# Fine-tune YOLO11x on Kaggle (GPU) instead of this laptop

Kaggle gives free GPU notebooks (T4 / P100, roughly 30 GPU-hours a week, sessions of about 9 to 12 hours). The
laptop only packages, uploads and downloads; the training itself runs on Kaggle under **your** account.

## One-time setup (only you can do this: it needs your Kaggle login)
1. Create/verify a Kaggle account. **Phone verification is required** to enable GPU and Internet in notebooks.
2. `pip install kaggle`
3. kaggle.com → profile → **Settings → API → Create New Token**. Save the downloaded `kaggle.json` to
   `%USERPROFILE%\.kaggle\kaggle.json` (a new-style token can go in `~/.kaggle/access_token` or the `KAGGLE_API_TOKEN` variable).
   That file is a password for your account: keep it private, never commit it, never paste it into chat.
4. `python kaggle_train/launch.py check` should now print `ready: yes`.

## Run
```bash
python kaggle_train/launch.py prepare                  # builds kaggle_train/_upload/ (offline; nothing is sent)
python kaggle_train/launch.py all                      # prepare + upload (3 PRIVATE datasets) + start the GPU notebook
python kaggle_train/launch.py status                   # running / complete / error
python kaggle_train/launch.py pull                     # downloads best.pt, last.pt, results.csv, run_summary.json -> runs/kaggle/
# with real annotations (recommended): a YOLO export with images/, labels/ and classes.txt
python kaggle_train/launch.py all --annotated path/to/yolo_export
# options baked into the notebook:  --epochs 120 --imgsz 640 --batch 6 --time-hours 8 --task detect|segment
```
What gets uploaded (all private by default): `yolo11x.pt` (SHA-256 checked first), the small `detect` code package,
and the 1,865 training photos in `vetdata/` (CC-BY-4.0 Zenodo datasets: keep their attribution if you ever make the data public).

## What the notebook does (`train_yolo11x.py`)
1. refuses to run without a GPU (so it never silently trains YOLO11x on a Kaggle CPU for hours), and re-checks the weights' SHA-256;
2. **annotated mode** if an annotated dataset is attached: validates every label line, removes near-duplicate images, hash-splits train/val;
   **weak mode** otherwise: the COCO YOLO11x boxes one animal per photo and labels it with the folder's condition;
3. fine-tunes with `configs/yolo_train.yaml` (imgsz 640, multi-scale, SGD 0.01 → cosine, warm-up, mosaic/mixup/HSV, cls loss 1.0), capped at `time_hours` so
   the schedule finishes inside the session; halves the batch automatically on CUDA out-of-memory;
4. saves the weights **first**, then validation metrics (`run_summary.json`), then best-effort ONNX and INT8 LiteRT/`.tflite` exports (a failed export never loses the weights).

## Be clear about what you will get
- **Weak mode is not a lesion detector.** It learns "diseased animal" from whole-animal boxes, covers only photos where the COCO detector finds a whole animal
  (about 65 % of them: in the first real run 1,211 of 1,865 photos got a box, but only **30 of about 306 FMD photos**, because most FMD photos are mouth/hoof close-ups and pigs are not a COCO class; the notebook warns about classes with zero boxes),
  and its mAP says nothing about finding nodules, tick clusters, wounds or FMD lesions. Real lesion detection needs boxes or polygons drawn by a vet/annotator
  (CVAT, Label Studio or Roboflow, export as "YOLO"), attached with `--annotated`.
- **Not verified on Kaggle itself**: no Kaggle account was available while building this. Verified locally: the notebook script end to end on a CPU at toy
  scale (dataset build, training, checkpoint save, validation metrics, ONNX export), the launcher's bundle layout, safety checks and the exact
  `kaggle` commands it would run (recorded in tests). Not verified: the real Kaggle API calls (the CLI and metadata fields change between versions;
  if `kernels push` rejects the metadata, `kaggle_train/_kernel/kernel-metadata.json` is the file to adjust), GPU memory at batch 6 / 640 px (the batch halves
  automatically if it overflows), and the INT8 export (needs Linux, which Kaggle is; it fails on Windows).
- **Time**: unmeasured on Kaggle. A rough guess for a T4 is a few minutes per epoch for about a thousand 640 px images, so 120 epochs may not fit one session;
  `time_hours` makes Ultralytics compress the schedule to fit, and `last.pt` lets you continue in a second run.
- Ship `yolo11s`/`yolo11m` on phones, not `x` (see `mobile/README.md`).

## Models and jobs
`launch.py run --model yolo11x|yolo11n|both` trains either model on the SAME pseudo-labelled data (both are labelled by the COCO YOLO11x). `pull` fetches only the
weights, results and logs (a plain download of the notebook output would fetch thousands of dataset images one request at a time).
`prepare --trained models/<version>/best.pt` + `upload --only trained` + `run --model yolo11n-export` exports a fine-tuned model to LiteRT (FP32, static INT8, dynamic) and
scores each on the same validation split. Lessons that cost runs: unverified phone = no GPU and no internet; ultralytics ignores `batch` for a long list of paths (chunk it yourself);
`/kaggle/input` is read-only (export from a copy); a freshly created dataset must finish processing before a notebook is pushed.
