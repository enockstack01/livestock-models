# LivestockDx mobile (React Native / Expo)

Online/offline livestock disease scanner.

| Situation | Detection | Report |
|---|---|---|
| Online (Vision API answers `/ready` within 8 s) | **Server model** on the server | **Qwen** via the LLM API; server advice pack if the LLM fails |
| Offline | **YOLO11n on the phone** | bundled advice pack (same wording as the server fallback, parity-tested) |

Offline scans are saved in a SQLite queue labelled "Preliminary". On app start, on returning to the foreground, and every 30 s while photos wait, they are re-sent to the server model with the phone's result attached.
The Queue tab shows both results and flags disagreements. Failures back off (30 s, doubling, max 1 h, 8 attempts); a 401 keeps the photo and asks you to sign in again.
Set the two API URLs and the access token in the Settings tab (defaults are placeholders: `https://livestock-vision.api.agricoders.com`, `https://llm.api.agricoders.com`).

Layout: `src/core/` pure TypeScript logic (scan, queue, guidance, YOLO decode; 31 Node tests incl. Python parity), `src/api/http.ts` fetch adapter, `src/store/storage.ts` SQLite, `src/ml/` on-device model, `src/screens/` UI.
**The screens, the storage and HTTP adapters and `App.tsx` type-check but have never been run on a phone or emulator.** Expect runtime issues on first launch (camera, TFLite loading, SQLite).
Dependencies are declared in `package.json` (with a lockfile). `npm install && npm run typecheck && npm test` passes: the whole app, including the screens, type-checks under TypeScript strict mode.

The on-device part works with **no internet** after install: the model, the disease knowledge base and all image processing live on the phone.

```
photo ─► expo-image-manipulator (native resize)
      ─► jpeg-js decode ─► letterbox + normalise      src/core/preprocess.ts  (pure TS, unit-tested)
      ─► react-native-fast-tflite (INT8 YOLO11)       src/ml/detector.ts
      ─► decode + per-class NMS + un-letterbox        src/core/yolo.ts        (pure TS, unit-tested)
      ─► labels ─► local SQLite (expo-sqlite)         src/db/diseases.ts      what it is / what to do now / report?
```

## Build it
```bash
cd mobile
npm install                       # or: npx expo install  (keeps SDK-compatible versions)
npm test                          # 9 unit tests of the pure-TypeScript core (Node, no device needed)
npx tsc --noEmit                  # type-check
npx expo prebuild                 # generates android/ and ios/ (applies the fast-tflite + camera plugins)
npx expo run:android              # or run:ios (macOS + Xcode)
```
`react-native-fast-tflite` is native code: **Expo Go cannot run it**; use a development build (commands above, or EAS Build).
Without a model the app still starts, shows the offline disease library, and says detection is unavailable.

## Install a trained model
The app ships with a **development model** (`yolo11n-weak-v1`, see below). There is no trained *lesion* detector yet: no annotated lesion data exists (see the top-level README).
LiteRT export only works on Linux/macOS, so it runs on Kaggle (Windows cannot do it):
```bash
python kaggle_train/launch.py run --model yolo11n                       # fine-tune
python kaggle_train/launch.py pull --model yolo11n                      # -> runs/kaggle/yolo11n/ (best.pt, run_summary.json)
python -m livestock_vision.register --run-dir runs/kaggle/yolo11n --version yolo11n-<name>-v2
python kaggle_train/launch.py prepare --trained models/yolo11n-<name>-v2/best.pt
python kaggle_train/launch.py upload --only trained && python kaggle_train/launch.py run --model yolo11n-export
python kaggle_train/launch.py pull --model yolo11n-export              # FP32 / static INT8 / dynamic .tflite + mAP of each
```
Install the **dynamic (`w8a32`)** file as `mobile/assets/models/model.tflite` and write `model.json` as in the current one (labels in the model's class order,
`input_layout: nchw`, `output_layout: channels_first`, `input_dtype: float32`, `imgsz: 640`). Prefer it over static INT8 unless a phone benchmark shows INT8 is needed for speed:
INT8 distorted confidence scores in the check below. **Check one real photo after installing**; if boxes look scrambled, the layout fields in `model.json` are wrong.

## Size and speed: do not ship the server model to phones
The server model is large (about 110 MB FP32, about 57 MB INT8). It is the most accurate but, even
quantised, expect seconds per photo on a mid-range phone and high memory use. For the phone use the small model
(`yolo11n`, 3 MB), or a mid-size one if you need more accuracy; keep the large model on the server.

## Offline knowledge base
`src/data/diseases.json` is loaded into SQLite on first launch (and again when `data_version` increases).
**It has not been reviewed by a veterinarian** (`reviewed_by_veterinarian: false`; the app shows a notice).
It deliberately contains no drug names or doses. Have a vet / your veterinary service review and localise it
(local disease names, reporting phone numbers, languages) before field use. FMD and LSD are reportable diseases:
the app tells users to contact the authorities.

## What was verified on the real exported files (2026-11-02)
`yolo11n-weak-v1` was fine-tuned on Kaggle and exported to LiteRT (`.tflite`) FP32, static INT8 and dynamic-range (w8a32). Opening the real files showed:
- **Input is planar `[1, 3, 640, 640]` float32 (NCHW)**, not the interleaved NHWC most TFLite models use. The app's first version built NHWC, which would have
  produced plausible-looking garbage without any error. `rgbaToTensor(..., 'nchw')` and `model.json` `input_layout` now handle it (unit-tested).
- **Output is `[1, 7, 8400]` float32 (channels-first)** even for the INT8 file; **boxes are normalised 0..1** in the TFLite output (pixels in ONNX). The decoder detects both.
- The app's real TypeScript preprocessing + the real `.tflite` files + the app's real decoder reproduced Ultralytics' detections on 8 of 8 test photos:

| Variant | Size | Detections matched | Mean confidence error | mAP50-95 (191 val photos) |
|---|---|---|---|---|
| FP32 | 10.7 MB | 8/8 | 0.002 | 0.456 |
| **w8a32 (installed)** | **3.0 MB** | 8/8 | 0.005 | 0.469 |
| static INT8 | 3.1 MB | 8/8 (+1 extra box) | **0.167** | 0.421 |

  Static INT8 finds the boxes but shifts confidence scores a lot, which matters because the verdict thresholds assume calibrated confidence. The `.pt` model scored 0.517 mAP50-95
  on the same photos; the gap to FP32 is most likely the fixed square 640x640 input of exported models (unverified). The 191-photo validation set is small: differences of a few points are noise.
- The decoder is also tested on raw output from the real model (`tests/fixtures/yolo11n_onnx/`), identical to Ultralytics' NMS to 4 decimals.
- **The installed model is a development model**: trained on whole-animal boxes (healthy / lumpy skin disease / FMD, FMD with only 30 boxes). The Scan screen shows its warning. It does not find lesions and is not for field use.

## Not built / not verified
- **Never run on a phone or emulator** in this repository's development environment: only the pure-TypeScript core is
  tested (`npm test`). Camera, TFLite loading, delegates and the UI need testing on real devices.
- No crop (ViT) mode, no live video-frame detection (it would use `react-native-vision-camera` frame processors),
  no multilingual UI.
- Quantised accuracy must be re-measured on the quantised model before release.
