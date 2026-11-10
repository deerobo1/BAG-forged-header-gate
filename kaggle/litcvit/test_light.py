"""Tests for the lightweight models and the counterfactual loss. Run: pytest kaggle/litcvit/test_light.py"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import light as T  # noqa: E402
import litcvit as L  # noqa: E402


def packet(ttl, window, seq, payload=b"data"):
    ip = struct.pack("!BBHHHBBH8x", 0x45, 0, 40 + len(payload), 7, 0x4000, ttl, 6, 0xBEEF)
    tcp = struct.pack("!4xIIBBHHH", seq, seq + 1, 5 << 4, 0x18, window, 0xCAFE, 0)
    return (ip + tcp).ljust(L.HEADER, b"\0") + payload.ljust(L.ROW - L.HEADER, b"\0")


def flows(*specs):
    rows = [packet(*s).ljust(L.ROW, b"\0") + bytes(L.ROW * (L.PACKETS - 1)) for s in specs]
    return torch.from_numpy(np.frombuffer(b"".join(rows), np.uint8).reshape(len(specs), L.PACKETS, L.ROW).copy())


TTL, WINDOW, SEQ = 8, 34, 24  # byte offsets in a packet with a 20-byte IP header


def test_sequence_numbers_can_be_left_out_of_the_mask():
    x = flows((64, 1000, 5))
    full, stack = L.shortcut_fields(x)[0, 0], L.shortcut_fields(x, sequence=False)[0, 0]
    assert full[SEQ:SEQ + 8].all() and not stack[SEQ:SEQ + 8].any()
    assert stack[TTL] and stack[WINDOW] and torch.equal(full & ~stack, torch.isin(torch.arange(L.ROW), torch.arange(SEQ, SEQ + 8)))


def test_counterfactual_takes_the_donors_fields_and_keeps_the_rest():
    x = flows((128, 8192, 100, b"malware"), (32, 5792, 200, b"benign!"))
    fields = T.stack_fields(x)
    c = T.counterfactual(x, fields, torch.tensor([1, 0]))
    assert c[0, 0, TTL] == 32 and c[1, 0, TTL] == 128
    assert c[0, 0, WINDOW] == x[1, 0, WINDOW] and c[0, 0, WINDOW + 1] == x[1, 0, WINDOW + 1]
    assert torch.equal(c[0, 0, SEQ:SEQ + 8], x[0, 0, SEQ:SEQ + 8])  # own sequence numbers
    assert torch.equal(c[0, 0, L.HEADER:], x[0, 0, L.HEADER:])  # own payload
    assert torch.equal(c[0, 0, 2:4], x[0, 0, 2:4])  # own length


def test_counterfactual_leaves_missing_packets_alone():
    a = flows((128, 8192, 100))
    b = flows((32, 5792, 200))
    b[0, 0] = 0  # the donor has no first packet
    x = torch.cat([a, b])
    c = T.counterfactual(x, T.stack_fields(x), torch.tensor([1, 0]))
    assert torch.equal(c[0], x[0]) and torch.equal(c[1], x[1])


def test_donors_come_from_other_classes():
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    d = T.donors_of(y, torch.Generator().manual_seed(0))
    assert (y[d] != y).all()


def unit(*v):
    return F.normalize(torch.tensor(v, dtype=torch.float32), dim=1)


def test_loss_punishes_an_encoder_that_reads_the_header():
    # Two flows of different classes, each lending its header to the other's counterfactual.
    y, donors = torch.tensor([0, 1]), torch.tensor([1, 0])
    z_x = unit([1.0, 0.0], [0.0, 1.0])
    reads_content = T.counterfactual_nce(unit([1.0, 0.0], [0.0, 1.0]), z_x, y, donors)
    reads_header = T.counterfactual_nce(unit([0.0, 1.0], [1.0, 0.0]), z_x, y, donors)
    assert reads_content < 0.05 and reads_header > 3


def test_donor_counts_more_than_other_negatives():
    y, donors = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])
    z_x = unit([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    z_c = unit([1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0])  # each halfway to its donor
    assert T.counterfactual_nce(z_c, z_x, y, donors, beta=8.0) > T.counterfactual_nce(z_c, z_x, y, donors, beta=1.0)


def test_same_class_flows_are_no_negatives():
    y, donors = torch.tensor([0, 0]), torch.tensor([1, 0])
    z = unit([1.0, 0.0], [0.0, 1.0])
    assert T.counterfactual_nce(z, z, y, donors).abs() < 1e-5


def test_loss_is_differentiable_with_the_infinite_log_weights():
    y, donors = torch.tensor([0, 0, 1, 1]), torch.tensor([2, 3, 0, 1])
    z_x = F.normalize(torch.randn(4, 8, generator=torch.Generator().manual_seed(0)), dim=1).requires_grad_()
    z_c = F.normalize(torch.randn(4, 8, generator=torch.Generator().manual_seed(1)), dim=1).requires_grad_()
    T.counterfactual_nce(z_c, z_x, y, donors).backward()
    assert torch.isfinite(z_x.grad).all() and torch.isfinite(z_c.grad).all() and z_c.grad.abs().sum() > 0


def test_mamba_scan_is_causal():
    ssm = T.SelectiveSSM(8)
    u = torch.randn(1, 12, 8, generator=torch.Generator().manual_seed(0))
    changed = u.clone()
    changed[0, 8:] += 1.0
    with torch.no_grad():
        a, b = ssm(u), ssm(changed)
    assert torch.allclose(a[0, :8], b[0, :8]) and not torch.allclose(a[0, 8:], b[0, 8:])


def test_bidirectional_block_sees_both_ways():
    block = T.BiMambaBlock(8)
    u = torch.randn(1, 12, 8, generator=torch.Generator().manual_seed(0))
    changed = u.clone()
    changed[0, 6] += torch.randn(8, generator=torch.Generator().manual_seed(1))  # a constant would vanish in LayerNorm
    with torch.no_grad():
        a, b = block(u), block(changed)
    assert (a - b)[0, 0].abs().max() > 1e-5 and (a - b)[0, -1].abs().max() > 1e-5


def test_masked_model_ignores_shortcut_fields():
    net = T.FlowNet("vim", 3, masked=True).eval()
    x, y = flows((128, 8192, 100)), flows((32, 5792, 200))
    with torch.no_grad():
        assert torch.allclose(net(x), net(y))


def test_evasion_counts_only_attacked_flows_detected_as_recorded():
    # Classes: 0 Benign, 1 Dridex. Flows 0-1 benign; 2-5 malware; flow 5 had no donor, so it isn't attacked.
    data = {"y_test": np.array([0, 0, 1, 1, 1, 1]), "classes": ["Benign", "Dridex"],
            "clean_pred": np.array([0, 0, 1, 1, 0, 1]),  # flow 4 was already missed as recorded
            "attacked": {"mimic_stack": np.array([False, False, True, True, True, False])}}
    z = np.zeros((6, 2))
    z[np.arange(6), [0, 0, 0, 1, 0, 1]] = 1.0  # flow 2 now called Benign, flow 3 still caught
    row = T.score(data, "mimic_stack", z)
    assert row["attacked"] == 3
    assert row["detected_clean"] == pytest.approx(66.7)
    assert row["evaded"] == 50.0  # of flows 2 and 3, one got through; flow 4 doesn't count
    assert "evaded" not in T.score(data, "clean", z)


def test_forge_gives_malware_flows_a_benign_header_and_leaves_benign_flows_out():
    # Class 0 is Benign: TTL 32. Classes 1 and 2 are malware: TTL 128.
    x = flows((32, 5792, 1, b"benign"), (128, 8192, 2, b"mal-one"), (128, 8192, 3, b"mal-two"), (32, 5840, 4, b"benign"))
    y = torch.tensor([0, 1, 2, 0])
    forged, bad = T.forge(x, T.stack_fields(x), y, torch.tensor([True, False, False]), torch.Generator().manual_seed(0))
    assert bad.tolist() == [1, 2]
    assert (forged[:, 0, TTL] == 32).all()
    window = forged[:, 0, WINDOW].int() * 256 + forged[:, 0, WINDOW + 1].int()
    assert set(window.tolist()) <= {5792, 5840}
    assert torch.equal(forged[0, 0, L.HEADER:], x[1, 0, L.HEADER:])  # each keeps its own payload
    assert torch.equal(forged[1, 0, L.HEADER:], x[2, 0, L.HEADER:])
    assert torch.equal(forged[0, 0, SEQ:SEQ + 8], x[1, 0, SEQ:SEQ + 8])  # and its own sequence numbers


def test_forge_needs_both_kinds_of_flow():
    x = flows((128, 8192, 1), (128, 8192, 2))
    forged, bad = T.forge(x, T.stack_fields(x), torch.tensor([1, 2]), torch.tensor([True, False, False]), torch.Generator())
    assert len(forged) == 0 and len(bad) == 0


def test_one_sided_loss_teaches_that_a_benign_header_clears_nothing():
    # Benign and malware flows differ in the header and in the payload. Trained plainly, a model may
    # use either; a forged header must not turn a malware payload benign.
    specs = [(32, 5792, i, b"good" * 20) for i in range(32)] + [(128, 8192, i, b"evil" * 20) for i in range(32)]
    x, y = flows(*specs).numpy(), np.array([0] * 32 + [1] * 32)
    attacked = flows(*[(32, 5792, i, b"evil" * 20) for i in range(8)]).numpy()
    caught = {}
    for loss in ("ce", "oneside"):
        torch.manual_seed(0)
        net = T.FlowNet("shufflenet", 2)
        T.fit(net, x, y, loss, seed=0, epochs=60, batch=16, benign=np.array([True, False]))
        assert (T.logits_of(net, x).argmax(1) == y).all(), loss
        caught[loss] = (T.logits_of(net, attacked).argmax(1) == 1).mean()
    assert caught["ce"] < 0.5 and caught["oneside"] == 1.0


@pytest.mark.parametrize("loss", ["ce", "mask", "rand", "swap", "both", "cfnce", "cfnce1", "oneside"])
def test_every_loss_trains_and_runs(loss):
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    x = torch.cat([flows((60 + int(c) * 20, 1000 + int(c), 10 + i)) for i, c in enumerate(y)]).numpy()
    net = T.FlowNet("vim", 3, masked=loss == "mask")
    value = T.fit(net, x, y, loss, seed=0, epochs=1, batch=4, benign=np.array([True, False, False]))
    assert np.isfinite(value)
    assert T.logits_of(net, x).shape == (8, 3)


# ---------------------------------------------------------------- the gate

def test_gate_never_lets_a_flow_the_content_model_calls_malware_pass_as_benign():
    benign = torch.tensor([True, False, False, False])
    gen = torch.Generator().manual_seed(0)
    content = torch.randn(4096, 4, generator=gen)
    full = 10 * torch.randn(4096, 4, generator=gen)  # any full model's answer, as an attacker might drive it
    out = T.gate(full, content, benign).argmax(1)
    convicted = ~benign[content.argmax(1)]
    assert convicted.sum() > 1000
    assert not benign[out[convicted]].any()


def test_gate_holds_with_several_benign_classes_and_passes_a_flow_both_call_benign():
    benign = torch.tensor([True, True, False])
    content = torch.tensor([[0.0, 1.0, 2.0], [5.0, 1.0, 0.0]])
    full = torch.tensor([[9.0, 9.0, -9.0], [1.0, 7.0, 0.0]])
    assert T.gate(full, content, benign).argmax(1).tolist() == [2, 0]


@pytest.mark.parametrize("name", ["shufflenet", "vim"])
def test_a_gated_network_keeps_its_answer_on_flows_it_convicts_whatever_their_shortcut_fields_hold(name):
    torch.manual_seed(0)
    benign = np.array([True, False, False])
    net = T.GatedNet(name, 3, benign).eval()
    x = torch.randint(0, 256, (64, L.PACKETS, L.ROW), dtype=torch.uint8)
    x[:, :, 0], x[:, :, 9], x[:, :, 32] = 0x45, 6, 5 << 4  # IPv4, TCP, no options
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        full, content = net.views(x)
        convicted = ~net.benign[content.argmax(1)]
        for _ in range(3):
            changed = L.randomize(x, T.all_fields(x), gen)
            assert not torch.equal(changed, x)
            full2, content2 = net.views(changed)
            assert torch.equal(content2, content), "the content pass read a shortcut field"
            assert not torch.equal(full2, full), "the full pass ignored the shortcut fields"
            assert not net.benign[net(changed).argmax(1)][convicted].any()
    assert convicted.any()


def test_a_gated_networks_answer_on_zeroed_flows_is_its_content_answer():
    torch.manual_seed(0)
    net = T.GatedNet("shufflenet", 3, np.array([True, False, False])).eval()
    x = torch.randint(0, 256, (64, L.PACKETS, L.ROW), dtype=torch.uint8)
    x[:, :, 0], x[:, :, 9], x[:, :, 32] = 0x45, 6, 5 << 4  # IPv4, TCP, no options
    with torch.no_grad():
        _, content = net.views(x)
        assert torch.equal(net(L.perturb(x, "zero", None)), content)  # gate_report reads the certified share from this


def test_gated_losses_reach_the_network_from_both_views():
    torch.manual_seed(0)
    x = flows((64, 1000, 5, b"evil"), (64, 1000, 5, b"good"), (128, 2000, 9, b"evil"), (128, 2000, 9, b"good"))
    y = torch.tensor([1, 0, 1, 0])
    weight = torch.ones(2)
    benign = np.array([True, False])
    for loss in ("gated", "views"):
        net = T.GatedNet("shufflenet", 2, benign)
        net.train()
        value = T.step_loss(net, loss, x, y, T.all_fields(x), weight, None, torch.as_tensor(benign))
        value.backward()
        assert value.item() > 0 and net.head.weight.grad.abs().sum() > 0
