import os
import shutil
import argparse
import numpy as np
import cv2
import tensorflow as tf

# ==========================================
# 1. INTERACTIVE IMAGE LABELING (OpenCV)
# ==========================================
def label_images(raw_dir="raw_images", output_dir="dataset"):
    """
    Opens photos sequentially in a preview window.
    Press 'u' to sort into dataset/up
    Press 'd' to sort into dataset/down
    Press 's' to skip
    Press 'q' to quit
    """
    up_path = os.path.join(output_dir, "up")
    down_path = os.path.join(output_dir, "down")
    os.makedirs(up_path, exist_ok=True)
    os.makedirs(down_path, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"Error: Folder '{raw_dir}' does not exist. Create it and add your unlabelled photos.")
        return

    images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images in '{raw_dir}'.")
    print("Controls: [u] = UP, [d] = DOWN, [s] = SKIP, [q] = QUIT\n")

    for img_name in images:
        src_path = os.path.join(raw_dir, img_name)
        img = cv2.imread(src_path)
        if img is None:
            continue

        h, w, _ = img.shape
        display_img = cv2.resize(img, (600, int(600 * h / w)))
        cv2.imshow("Labeling - [u]=UP, [d]=DOWN, [s]=SKIP, [q]=QUIT", display_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('u'):
            shutil.move(src_path, os.path.join(up_path, img_name))
            print(f"Sorted -> UP: {img_name}")
        elif key == ord('d'):
            shutil.move(src_path, os.path.join(down_path, img_name))
            print(f"Sorted -> DOWN: {img_name}")
        elif key == ord('s'):
            print(f"Skipped: {img_name}")
        elif key == ord('q'):
            print("Labeling session exited.")
            break

    cv2.destroyAllWindows()


# ==========================================
# 2. MODEL TRAINING & TFLITE EXPORT
# ==========================================
def train_and_export(dataset_dir="dataset", tflite_path="webbing_model.tflite", epochs=15):
    """
    Fine-tunes a MobileNetV3 model on the dataset and exports a compressed .tflite model.
    """
    img_size = (224, 224)
    batch_size = 16

    print("Loading dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Classes identified (alphabetical): {class_names}")  # Index 0: down, Index 1: up

    # Data Augmentation to learn lighting/texture variations
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])

    # Transfer Learning with MobileNetV3 Small (Lightweight for Mobile)
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("\n--- Training Model ---")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    print("\n--- Converting Model to TFLite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic range quantization
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Success! Model exported to '{tflite_path}'.")


# ==========================================
# 3. TFLITE INFERENCE
# ==========================================
def run_inference(image_path, tflite_path="webbing_model.tflite"):
    """
    Runs local prediction on a single test image using the TFLite runtime.
    """
    if not os.path.exists(tflite_path):
        print(f"Error: Model file '{tflite_path}' not found. Run training first.")
        return

    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Read and preprocess test image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Index 0 = 'down', Index 1 = 'up'
    label = "UP" if raw_output > 0.5 else "DOWN"
    confidence = raw_output if raw_output > 0.5 else (1 - raw_output)

    print(f"\nInference Result for '{image_path}':")
    print(f"Orientation: {label}")
    print(f"Confidence:  {confidence * 100:.2f}%")


# ==========================================
# CLI ARGUMENT PARSER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slackline Webbing Orientation ML Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["label", "train", "infer"],
                        help="Choose workflow step: 'label', 'train', or 'infer'")
    parser.add_argument("--image", type=str, help="Path to a test image for inference (required for --mode infer)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")

    args = parser.parse_args()

    if args.mode == "label":
        label_images()
    elif args.mode == "train":
        train_and_export(epochs=args.epochs)
    elif args.mode == "infer":
        if not args.image:
            print("Error: Please specify an image using --image path/to/image.jpg")
        else:
            run_inference(args.image)