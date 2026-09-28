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

      let outputTensor = output;
      if (!(output instanceof tf.Tensor)) {
        const firstKey = Object.keys(output)[0];
        outputTensor = output[firstKey];
      }
      if (!outputTensor || typeof outputTensor.data !== "function") {
        throw new Error("Model returned no usable output tensor.");
      }

      const raw = (await outputTensor.data())[0];
      if (!Number.isFinite(raw) || raw < 0 || raw > 1) {
        throw new Error(`Model returned invalid confidence: ${raw}`);
      }
      self.postMessage({
        requestId: data.requestId,
        raw,
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
