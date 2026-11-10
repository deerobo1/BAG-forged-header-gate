# Running the sprint on Kaggle

Files to upload: `C:\data\CNS_proj\kaggle\sprint_upload\` (litcvit-flows.zip, sprint-code.zip, sprint-weights.zip).

1. **Account:** Settings > Phone verification (needed for GPU and internet).
2. **Three private datasets:** Datasets > New Dataset, one per zip, Private. Kaggle unzips them on upload.
3. **Notebook:** Code > New Notebook.
   - Add Input: the three datasets.
   - Settings: Accelerator **GPU P100** (or T4 x2); Internet **on**.
4. **One cell:**

   ```python
   !pip install -q timm==1.0.30
   import pathlib
   JOB = "seed0"   # one notebook run each: seed0, time, seed1, seed2
   job = next(pathlib.Path("/kaggle/input").rglob("kaggle_job.py"))
   !python -u {job} {JOB} 2>&1 | tee /kaggle/working/{JOB}.log
   ```

5. **Run in the background:** Save Version > **Save & Run All (Commit)**. You can close the tab.
6. **Run two at once.** Kaggle allows 2 GPU sessions. Make a second notebook the same way with `JOB = "time"`. When those finish, run `seed1` and `seed2`.
7. **Cost:** about 5-6 h per job, about 22 GPU-hours for all four, inside the 30 h a week.
8. **When a job finishes:** open its version > Output > download everything. Unzip it to `C:\data\CNS_proj\kaggle\sprint_output_<JOB>` and tell me. It contains results/, models/, logits/ and the log.

If a job stops with "attach the sprint-weights dataset" or "no GPU", fix that setting and run it again.
