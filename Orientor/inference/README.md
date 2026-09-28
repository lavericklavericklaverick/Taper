# Webbing Auto-Aligner

## Files and deployment

- The GitHub Pages app is `index.html` in this folder.
- Its TFLite model is `webbing_model.tflite` in the same folder.
- The live page is <https://lavericklavericklaverick.github.io/Taper/Orientor/inference/>.
- Commit and push changes to `main`, then wait for the Pages deployment in GitHub Actions to finish before testing the live page.
- The page displays its version under the title. Bump it when publishing a page change, then open the live page with `?v=<version>` to avoid a stale browser cache.

## Before publishing

1. Edit the tracked file in `Orientor/inference/`; do not paste a merge-conflict result into the live page.
2. Search `index.html` for `<<<<<<<`, `=======`, and `>>>>>>>`. None may remain: conflict markers can appear as page text and invalidate the JavaScript.
3. Check that the inline JavaScript parses and run `git diff --check`.
4. After deployment, confirm the live page shows the expected version, model-ready status, and camera preview before testing BLE or Auto Align.

## Startup and alignment

1. Open the live page over HTTPS.
2. Wait for the model status to say it is ready.
3. Start the camera and connect to `ESP32C6_Stepper`.
4. Test one motor step before starting Auto Align.
5. During alignment, read the status below the model progress bar: it reports camera/model inference, the predicted orientation and confidence, each step, and any missing prerequisite or error.

Auto Align steps only for a confident DOWN prediction (`raw <= 0.30`) and stops for a confident UP prediction (`raw >= 0.70`). Predictions between those thresholds are reported but do not move the motor.

## Diagnosing Auto Align

- Confirm the page shows the version in `index.html`; the browser tab may still have an older deployment cached.
- The on-screen console keeps the latest 80 messages so repeated predictions cannot grow the page indefinitely.
- Each alignment frame logs the camera dimensions, crop time, TFLite predict start and duration, and prediction/confidence.
- If the page stops responding, note the last visible frame message. A last message saying `calling TFLite predict` points to the synchronous model prediction; later messages narrow it to output reading or motor communication.
- After changing the page, repeat the pre-publish checks and wait for GitHub Pages deployment before testing.
