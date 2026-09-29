importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.10.0/dist/tf-core.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@4.10.0/dist/tf-backend-cpu.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js"
);

const WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/";
let tfliteModel = null;

function disposeOutput(output) {
  if (output instanceof tf.Tensor) {
    output.dispose();
  } else if (output) {
    for (const tensor of Object.values(output)) {
      if (tensor && typeof tensor.dispose === "function") tensor.dispose();
    }
  }
}

self.addEventListener("message", async ({ data }) => {
  if (data.type === "load-model") {
    let modelUrl;
    try {
      self.postMessage({ type: "runtime-status", message: "Starting background TFLite runtime." });
      await tf.ready();
      tflite.setWasmPath(WASM_BASE_URL);
      modelUrl = URL.createObjectURL(new Blob([data.modelBuffer], { type: "application/octet-stream" }));
      tfliteModel = await tflite.loadTFLiteModel(modelUrl, { numThreads: 1 });
      self.postMessage({ requestId: data.requestId, ok: true });
    } catch (err) {
      tfliteModel = null;
      self.postMessage({ requestId: data.requestId, error: `${err.name}: ${err.message}` });
    } finally {
      if (modelUrl) URL.revokeObjectURL(modelUrl);
    }
    return;
  }

  if (data.type === "predict") {
    let inputTensor;
    let output;
    try {
      if (!tfliteModel) throw new Error("TFLite model is not ready.");
      inputTensor = tf.tensor(new Float32Array(data.pixels), [1, 224, 224, 3], "float32");
      const inferenceStartedAt = performance.now();
      output = tfliteModel.predict(inputTensor);

      const tensors = output instanceof tf.Tensor ? [output] : Object.values(output || {});
      if (!tensors.length || tensors.some(tensor => typeof tensor.data !== "function")) {
        throw new Error("Model returned no usable output tensors.");
      }

      const values = await Promise.all(tensors.map(async tensor => Array.from(await tensor.data())));
      const statusScores = values.find(value => value.length === 3);
      const angleVector = values.find(value => value.length === 2);
      if (statusScores && angleVector) {
        if (statusScores.some(score => !Number.isFinite(score) || score < 0 || score > 1)) {
          throw new Error(`Model returned invalid status scores: ${statusScores.join(", ")}`);
        }
        const statusIndex = statusScores.indexOf(Math.max(...statusScores));
        const statuses = ["target", "wrong", "none"];
        const angleDeg = (Math.atan2(angleVector[0], angleVector[1]) * 180 / Math.PI + 360) % 360;
        self.postMessage({
          requestId: data.requestId,
          status: statuses[statusIndex],
          confidence: statusScores[statusIndex],
          statusScores,
          angleDeg,
          inferenceMs: performance.now() - inferenceStartedAt
        });
        return;
      }

      const raw = values.length === 1 && values[0].length === 1 ? values[0][0] : NaN;
      if (!Number.isFinite(raw) || raw < 0 || raw > 1) {
        throw new Error("Unsupported model outputs; expected status[3] and angle[2].");
      }
      self.postMessage({
        requestId: data.requestId,
        raw,
        legacy: true,
        inferenceMs: performance.now() - inferenceStartedAt
      });
    } catch (err) {
      self.postMessage({ requestId: data.requestId, error: `${err.name}: ${err.message}` });
    } finally {
      if (inputTensor) inputTensor.dispose();
      disposeOutput(output);
    }
  }
});
