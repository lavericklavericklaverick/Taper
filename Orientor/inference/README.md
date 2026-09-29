# Webbing Auto-Aligner

## Files and deployment

- The GitHub Pages app is `index.html` in this folder.
- Its TFLite model is `webbing_model.tflite` in the same folder.
- TFLite runs in `inference-worker.js` so synchronous model inference does not block the page UI.
- The live page is <https://lavericklavericklaverick.github.io/Taper/Orientor/inference/>.
- Commit and push changes to `main`, then wait for the Pages deployment in GitHub Actions to finish before testing the live page.
- The page displays its version under the title. Bump it when publishing a page change, then open the live page with `?v=<version>` to avoid a stale browser cache.

## Before publishing

1. Edit the tracked file in `Orientor/inference/`; do not paste a merge-conflict result into the live page.
2. Search `index.html` for `<<<<<<<`, `=======`, and `>>>>>>>`. None may remain: conflict markers can appear as page text and invalidate the JavaScript.
3. Check that the inline JavaScript and `inference-worker.js` parse, then run `git diff --check`.
4. After deployment, confirm the live page shows the expected version, model-ready status, and camera preview before testing orientation checks or BLE.

## Startup and orientation check

1. Open the live page over HTTPS.
2. Wait for the model status to say it is ready.
3. Start the camera.
4. Tap **Check Orientation**. With the multi-task model, the status reports correct webbing and its angle from 0° to 360°, wrong webbing, or no webbing. The existing binary model remains supported as a legacy fallback.
5. Orientation checks do not connect to BLE or move the motor. Connect to `ESP32C6_Stepper` only if you want to use **Test Step ('R')**.

## Diagnosing orientation checks

- Confirm the page shows the version in `index.html`; the browser tab may still have an older deployment cached.
- The on-screen status reports the latest orientation result or an error; the debug console keeps the latest 80 messages.
- TFLite inference runs in a Web Worker with one runtime thread to keep the page responsive on phones. The status and debug console remain on the page thread.
- `orientation_pipeline.py` trains and runs the new model on center-cropped, 224x224 RGB images represented as float32 pixel values from 0 to 255. The browser uses the same input shape, channel order, value range, and integer-centered square crop; MobileNetV3 preprocessing is included in the model graph.
- The browser loads `Orientor/inference/webbing_model.tflite`. Replace that file with the newly exported `webbing_orientation_model.tflite` only after training succeeds.
- The new model has a three-way status output (`target`, `wrong`, or `none`) and a circular angle output encoded as sine and cosine. Angle is reported only when `target` has the highest score.
- The browser worker also accepts the original one-score UP/DOWN model so the checked-in model keeps working until a new model has been trained.
- Orientation checks require only a ready model and an active camera; BLE is needed only for the separate manual motor-step test.
- After changing the page, repeat the pre-publish checks and wait for GitHub Pages deployment before testing.

## Multi-task training pipeline

Install the Python dependencies with `python -m pip install -r requirements.txt`.

The pipeline learns two related tasks:

1. Classify the frame as `target` (the correct webbing), `wrong` (other webbing), or `none` (no webbing).
2. For `target` frames only, regress the circular angle from 0° to 360°. The angle is represented internally as sine and cosine so that 359° remains close to 0°.

Label whole videos in a CSV matching `video_manifest.example.csv`. Paths are relative to the CSV. `target` rows require an angle; `wrong` and `none` rows must leave it blank. Record target videos at consistent known angles, such as 0°, 20°, ..., 340°. Include varied lighting, backgrounds, distances, and several videos for each non-target class. Avoid mixing angles within one target video.

The split is performed by source video, not by frame, which prevents near-identical frames from the same video appearing in training and validation. Each status should have at least two source videos so it can appear in both sets.

Prepare frames without training:

```powershell
python orientation_pipeline.py prepare --manifest video_manifest.csv --overwrite-prepared
```

Train, validate, and export:

```powershell
python orientation_pipeline.py train --manifest video_manifest.csv --epochs 20 --overwrite-prepared
```

Run local TFLite inference:

```powershell
python orientation_pipeline.py infer --image test.jpg --tflite webbing_orientation_model.tflite
```

Training writes `webbing_orientation_model.tflite`, a JSON metadata sidecar, and validation metrics for status accuracy and circular angle mean absolute error. To deploy a successful model, copy it over `inference/webbing_model.tflite`, then publish the page. Use `--max-frames-per-video` for quick experiments and `--fps` to adjust frame sampling.
