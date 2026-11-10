"""How far does the gate hold? Attacks inside and outside its guarantee, on saved weights, on CPU.

Models: MobileNetV2 and ShuffleNet from kaggle/litcvit_ablate and kaggle/litcvit_gate (the one-sided loss), LitCVit and
its SA version from kaggle/litcvit_evasion, one-sided LitCVit from kaggle/litcvit_oneside, each as trained and gated
(light.gate) by the MobileNetV2 content model where its weights exist. Attacks (attacks.py,
on malware flows only, benign flows stay as recorded):
  mimic_stack, mimic_all  copying a benign flow's fields (light.test_sets)
  arbitrary               random values in the fields a stack may set as it likes
  arbitrary_any           the same with several draws per flow on a sample of flows: a flow counts as evaded if any
                          draw is called Benign, as an attacker who may probe the model would find it
  arbitrary_options       arbitrary, with random values in well-formed TCP options as well
  add_packets             a packet repeated twice, inside the five read
  resegment_1, _8         the first data in pieces of 1 or 8 bytes
  mimic_add, mimic_resegment  mimic_stack together with add_packets or resegment_1: neither header nor content left
The first five change only fields the content model never reads, so the guarantee covers them. The others don't.
evaded: of the flows the attack changed that the method detects as recorded, the share called Benign.

  python kaggle/litcvit/limits.py

Writes results/litcvit_limits.csv after each model. Models whose weights aren't downloaded yet are skipped, and models
already in the file aren't repeated, so running it again adds only what's new.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import attacks as A
import gate_report as G
import light as T

L = T.L
ROOT = Path(__file__).resolve().parent.parent.parent
KAGGLE = Path("C:/data/CNS_proj/kaggle")
ABLATE, EVASION = KAGGLE / "litcvit_output_ablate" / "preds", KAGGLE / "litcvit_output_evasion" / "models"
GATE_JOB, ONESIDE = KAGGLE / "litcvit_output_gate" / "preds", KAGGLE / "litcvit_output_oneside" / "models"
LIGHT = {"mobilenetv2": ("ce", "both", "cfnce", "oneside"), "shufflenet": ("both", "cfnce", "oneside")}
HEAVY = {"litcvit": EVASION, "litcvit-sa": EVASION, "litcvit_oneside": ONESIDE}
CONTENT = ("mobilenetv2", "mask")
DRAWS, SAMPLE = 8, 2000
BATCH = 128


def light_path(run, model, loss):
    return next((p for p in (folder / f"{run}_{model}_{loss}.pt" for folder in (ABLATE, GATE_JOB)) if p.exists()), None)


def light_model(run, model, loss, k):
    path = light_path(run, model, loss)
    if path is None:
        return None
    net = T.FlowNet(model, k, masked=loss == "mask")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return lambda images: T.logits_of(net, images, batch=BATCH)


def heavy_model(run, model):
    path = HEAVY[model] / f"{run}_{model}.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder = L.Encoder().eval()
    encoder.load_state_dict(state["encoder"])
    head = L.classifier_head(L.D, len(state["classes"])).eval()
    head.load_state_dict(state["head"])

    def logits(images):
        with torch.no_grad():
            return torch.cat([head(encoder.from_packets(encoder.packet_tokens(torch.as_tensor(images[i:i + BATCH]))))
                              for i in range(0, len(images), BATCH)]).numpy()
    return logits


def conditions_of(data, seed=0):
    images = data["conditions"]["clean"].reshape(-1, L.PACKETS, L.ROW)
    malware = ~np.isin(np.array(data["classes"])[data["y_test"]], list(T.BENIGN))
    rng = np.random.default_rng(seed)
    copied = data["conditions"]["mimic_stack"].reshape(images.shape)
    return {"mimic_stack": copied, "mimic_all": data["conditions"]["mimic_all"].reshape(images.shape),
            "arbitrary": A.arbitrary(images, rng, only=malware), "arbitrary_options": A.arbitrary(images, rng, only=malware, options=True),
            "add_packets": A.add_packets(images, only=malware), "resegment_1": A.resegment(images, 1, only=malware),
            "resegment_8": A.resegment(images, 8, only=malware), "mimic_add": A.add_packets(copied, only=malware),
            "mimic_resegment": A.resegment(copied, 1, only=malware)}


def answers(logits, content, gate, benign):
    """The method's answer on each flow, given the base model's and the content model's logits on it."""
    if gate == "none":
        return logits.argmax(1)
    return G.gate_decision(logits, content, benign) if gate == "decision" else G.gate_logits(logits, content, benign)


def row_of(run, name, gate, condition, y, benign, clean, pred, changed):
    m = changed & ~benign[y] & ~benign[clean]
    return {"dataset": run, "model": name, "gate": gate, "condition": condition, "macro_f1": round(100 * f1_score(y, pred, average="macro"), 1),
            "attacked": int(m.sum()), "evaded": round(100 * benign[pred[m]].mean(), 1) if m.any() else float("nan")}


def models_of(run):
    """The models with saved weights on a run: name -> (builder, its arguments up to the class count)."""
    models = {}
    for model, losses in LIGHT.items():
        for loss in losses:
            if light_path(run, model, loss) is not None:
                models[f"{model}-{loss}"] = (light_model, (run, model, loss))
    for model, folder in HEAVY.items():
        if (folder / f"{run}_{model}.pt").exists():
            models[model] = (heavy_model, (run, model))
    return models


def main(out=ROOT / "results" / "litcvit_limits.csv"):
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["dataset"], r["model"]) for r in rows}
    for run in L.RUNS:
        models = {name: m for name, m in models_of(run).items() if (run, name) not in done}
        if not models:
            continue
        data = T.test_sets(run, "random", 0)
        y, classes = data["y_test"], np.array(data["classes"])
        benign = np.isin(classes, list(T.BENIGN))
        images = data["conditions"]["clean"].reshape(-1, L.PACKETS, L.ROW)
        malware = ~benign[y]
        attacks = conditions_of(data)
        changed = {name: (x != images).any((1, 2)) for name, x in attacks.items()}
        content_net = light_model(run, *CONTENT, len(classes))
        content = None if content_net is None else {"clean": content_net(images), **{name: content_net(x) for name, x in attacks.items()}}
        rng = np.random.default_rng(1)
        sample = rng.permutation(np.flatnonzero(malware))[:SAMPLE]
        for name, (build, args) in models.items():
            net = build(*args, len(classes)) if build is light_model else build(*args)
            logits = {"clean": net(images), **{c: net(x) for c, x in attacks.items()}}
            attempts = [A.arbitrary(images[sample], np.random.default_rng(100 + d), options=True) for d in range(DRAWS)]
            reached = np.any([(x != images[sample]).any((1, 2)) for x in attempts], 0)
            draws = [net(x) for x in attempts]
            gates = ["none"] + ([] if content is None else ["decision"] + (["logits"] if "mobilenetv2" in name else []))
            for gate in gates:
                clean = answers(logits["clean"], None if content is None else content["clean"], gate, benign)
                base = row_of(run, name, gate, "clean", y, benign, clean, clean, np.zeros(len(y), bool))
                base["benign_recall"] = round(100 * benign[clean][~malware].mean(), 1)
                rows.append(base)
                for condition in attacks:
                    pred = answers(logits[condition], None if content is None else content[condition], gate, benign)
                    rows.append(row_of(run, name, gate, condition, y, benign, clean, pred, changed[condition]))
                called = np.stack([benign[answers(z, None if content is None else content["clean"][sample], gate, benign)] for z in draws])
                detected = reached & ~benign[clean[sample]]
                rows.append({"dataset": run, "model": name, "gate": gate, "condition": "arbitrary_any", "attacked": int(detected.sum()),
                             "evaded": round(100 * called[:, detected].any(0).mean(), 1) if detected.any() else float("nan")})
                print({k: v for k, v in rows[-1].items()}, flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    for value in ("evaded", "macro_f1"):
        print(f"\n--- {value}")
        print(df.pivot_table(index=["dataset", "model", "gate"], columns="condition", values=value, sort=False).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
