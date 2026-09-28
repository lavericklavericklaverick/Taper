import os
import argparse
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image  # Add to top imports

def run_inference(image_path, tflite_path="webbing_model.tflite"):
    if not os.path.exists(tflite_path):
        print(f"Error: Model file '{tflite_path}' not found. Train first.")
        return

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load via PIL to avoid Windows C++ backslash/null-byte (\0) string bugs
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_rgb = np.array(pil_img)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    # Center crop matching training logic
    img_processed = center_crop_and_resize(img_rgb, target_size=(224, 224))
    input_data = np.expand_dims(img_processed, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "UP" if raw_output > 0.5 else "DOWN"
    confidence = raw_output if raw_output > 0.5 else (1 - raw_output)

    print(f"\nInference Result for '{image_path}':")
    print(f"Orientation: {label}")
    print(f"Confidence:  {confidence * 100:.2f}%")

# ==========================================
# 1. ASPECT-RATIO PRESERVING CROP
# ==========================================
def center_crop_and_resize(img, target_size=(224, 224)):
    """
    Crops the center square of an image to preserve aspect ratio 
    before resizing, preventing thread pattern distortion.
    """
    h, w, _ = img.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped = img[top:top+min_dim, left:left+min_dim]
    return cv2.resize(cropped, target_size)


# ==========================================
# 2. MODEL TRAINING, FINE-TUNING & EXPORT
# ==========================================
def train_and_export(dataset_dir="dataset", tflite_path="webbing_model.tflite", epochs=20):
    img_size = (224, 224)
    batch_size = 16

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    print("Loading dataset with aspect-ratio preserving crops...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        crop_to_aspect_ratio=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        crop_to_aspect_ratio=True
    )

    class_names = train_ds.class_names
    print(f"Classes identified: {class_names}")  # [0: 'down', 1: 'up']

    # Data Augmentation: Crucial for texture variations
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomZoom(0.1)
    ])

    # Base Model
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze initially

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    # ------------------------------------------
    # PHASE 1: Train Dense Classifier Only
    # ------------------------------------------
    print("\n--- Phase 1: Training Top Classification Layer ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "best_webbing_model.keras", save_best_only=True, monitor="val_loss"
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    # ------------------------------------------
    # PHASE 2: Fine-Tuning Base Model Layers
    # ------------------------------------------
    print("\n--- Phase 2: Unfreezing Top Layers for Fine-Tuning ---")
    base_model.trainable = True
    
    # Freeze all except the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Low learning rate to prevent destroying pre-trained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    # ------------------------------------------
    # PHASE 3: Pre-Export Validation Check
    # ------------------------------------------
    print("\n--- Phase 3: Final Validation Check ---")
    best_model = tf.keras.models.load_model("best_webbing_model.keras")
    val_loss, val_acc = best_model.evaluate(val_ds)
    print(f"Best Validation Accuracy: {val_acc * 100:.2f}% | Best Validation Loss: {val_loss:.4f}")

    if val_acc < 0.85:
        print("\nWARNING: Model accuracy is below 85%. Consider gathering more images with varied angles before relying on this model.")

    # ------------------------------------------
    # PHASE 4: Export to TFLite
    # ------------------------------------------
    print("\n--- Converting Best Model to TFLite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Success! Model exported to '{tflite_path}'.")


# ==========================================
# 3. TFLITE INFERENCE WITH CENTER CROP
# ==========================================
def run_inference(image_path, tflite_path="webbing_model.tflite"):
    if not os.path.exists(tflite_path):
        print(f"Error: Model file '{tflite_path}' not found. Train first.")
        return

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply center crop matching training preprocessing
    img_processed = center_crop_and_resize(img_rgb, target_size=(224, 224))
    input_data = np.expand_dims(img_processed, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "UP" if raw_output > 0.5 else "DOWN"
    confidence = raw_output if raw_output > 0.5 else (1 - raw_output)

    print(f"\nInference Result for '{image_path}':")
    print(f"Orientation: {label}")
    print(f"Confidence:  {confidence * 100:.2f}%")


# ==========================================
# CLI ARGUMENT PARSER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slackline Webbing Orientation Trainer")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"],
                        help="Choose workflow step: 'train' or 'infer'")
    parser.add_argument("--image", type=str, help="Path to a test image for inference")
    parser.add_argument("--epochs", type=int, default=15, help="Fine-tuning epoch limit")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_export(epochs=args.epochs)
    elif args.mode == "infer":
        if not args.image:
            print("Error: Please specify an image using --image path/to/image.jpg")
        else:
            run_inference(args.image)