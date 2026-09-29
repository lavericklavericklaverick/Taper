import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 224
STATUS_NAMES = ("target", "wrong", "none")
STATUS_TO_INDEX = {name: index for index, name in enumerate(STATUS_NAMES)}


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    status: str
    angle_deg: Optional[float]


def circular_difference_degrees(actual, predicted):
    return abs((predicted - actual + 180.0) % 360.0 - 180.0)


def angle_to_vector(angle_deg):
    radians = math.radians(angle_deg)
    return np.array([math.sin(radians), math.cos(radians)], dtype=np.float32)


def vector_to_angle(vector):
    return math.degrees(math.atan2(float(vector[0]), float(vector[1]))) % 360.0


def read_video_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    records = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"video", "status", "angle_deg"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest columns must be: video,status,angle_deg")

        for line_number, row in enumerate(reader, start=2):
            status = row["status"].strip().lower()
            if status not in STATUS_TO_INDEX:
                raise ValueError(
                    f"Line {line_number}: status must be one of {', '.join(STATUS_NAMES)}"
                )
            video_path = Path(row["video"].strip())
            if not video_path.is_absolute():
                video_path = manifest_path.parent / video_path
            video_path = video_path.resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"Line {line_number}: video not found: {video_path}")

            angle_text = row["angle_deg"].strip()
            if status == "target":
                if not angle_text:
                    raise ValueError(f"Line {line_number}: target videos require angle_deg")
                angle = float(angle_text) % 360.0
            else:
                if angle_text:
                    raise ValueError(
                        f"Line {line_number}: angle_deg must be blank for {status} videos"
                    )
                angle = None
            records.append(VideoRecord(video_path, status, angle))

    if not records:
        raise ValueError("Manifest contains no videos")
    missing = set(STATUS_NAMES) - {record.status for record in records}
    if missing:
        raise ValueError(f"Manifest has no videos for: {', '.join(sorted(missing))}")
    insufficient = [
        status for status in STATUS_NAMES
        if sum(record.status == status for record in records) < 2
    ]
    if insufficient:
        raise ValueError(
            "At least two source videos are required for each status; add videos for: "
            + ", ".join(insufficient)
        )
    return records


def assign_video_splits(records, validation_fraction, seed):
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5")

    strata = {}
    for record in records:
        strata.setdefault(record.status, []).append(record)

    validation = set()
    for group in strata.values():
        ordered = sorted(
            group,
            key=lambda record: hashlib.sha256(
                f"{seed}:{record.path}".encode("utf-8")
            ).digest(),
        )
        validation_count = max(1, round(len(ordered) * validation_fraction)) if len(ordered) > 1 else 0
        validation.update(record.path for record in ordered[:validation_count])

    if not validation:
        raise ValueError(
            "A validation split could not be made. Supply at least two videos for one or more labels."
        )
    return {record.path: ("validation" if record.path in validation else "training") for record in records}


def extract_manifest_frames(
    manifest_path,
    output_dir="prepared_dataset",
    frames_per_second=3.0,
    validation_fraction=0.2,
    seed=42,
    max_frames_per_video=None,
    overwrite=False,
):
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    records = read_video_manifest(manifest_path)
    splits = assign_video_splits(records, validation_fraction, seed)
    output_dir = Path(output_dir).resolve()
    frames_dir = output_dir / "frames"
    index_path = output_dir / "frames.csv"

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if index_path.exists() and not overwrite:
        print(f"Using existing prepared dataset: {index_path}")
        return index_path
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_number, record in enumerate(records, start=1):
        capture = cv2.VideoCapture(str(record.path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {record.path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        frame_interval = max(1, round(source_fps / frames_per_second))
        video_id = hashlib.sha256(str(record.path).encode("utf-8")).hexdigest()[:10]
        destination = frames_dir / splits[record.path] / f"{video_number:03d}_{video_id}"
        destination.mkdir(parents=True, exist_ok=True)

        source_index = 0
        saved_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % frame_interval == 0:
                frame_path = destination / f"frame_{saved_count:05d}.jpg"
                if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    capture.release()
                    raise RuntimeError(f"Could not write frame: {frame_path}")
                rows.append(
                    {
                        "frame": str(frame_path.relative_to(output_dir)),
                        "split": splits[record.path],
                        "status": record.status,
                        "angle_deg": "" if record.angle_deg is None else f"{record.angle_deg:g}",
                        "source_video": str(record.path),
                    }
                )
                saved_count += 1
                if max_frames_per_video and saved_count >= max_frames_per_video:
                    break
            source_index += 1
        capture.release()
        if saved_count == 0:
            raise RuntimeError(f"No frames were extracted from: {record.path}")
        print(f"[{video_number}/{len(records)}] {record.path.name}: {saved_count} frames ({splits[record.path]})")

    with index_path.open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(
            index_file,
            fieldnames=("frame", "split", "status", "angle_deg", "source_video"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} frames at {index_path}")
    return index_path


def read_frame_index(index_path, split):
    index_path = Path(index_path).resolve()
    samples = []
    with index_path.open(newline="", encoding="utf-8-sig") as index_file:
        for row in csv.DictReader(index_file):
            if row["split"] != split:
                continue
            frame_path = (index_path.parent / row["frame"]).resolve()
            status_index = STATUS_TO_INDEX[row["status"]]
            is_target = row["status"] == "target"
            angle = angle_to_vector(float(row["angle_deg"])) if is_target else np.zeros(2, np.float32)
            samples.append((str(frame_path), status_index, angle, float(is_target)))
    if not samples:
        raise ValueError(f"Prepared dataset has no {split} frames")
    return samples


def make_dataset(samples, batch_size, training, seed):
    paths = np.array([sample[0] for sample in samples])
    statuses = np.array([sample[1] for sample in samples], dtype=np.int32)
    angles = np.stack([sample[2] for sample in samples])
    angle_weights = np.array([sample[3] for sample in samples], dtype=np.float32)
    status_weights = np.ones(len(samples), dtype=np.float32)
    if training:
        counts = np.bincount(statuses, minlength=len(STATUS_NAMES))
        class_weights = len(samples) / (len(STATUS_NAMES) * counts)
        status_weights = class_weights[statuses].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, statuses, angles, status_weights, angle_weights)
    )
    if training:
        dataset = dataset.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)

    def load_sample(path, status, angle, status_weight, angle_weight):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        shape = tf.shape(image)
        side = tf.minimum(shape[0], shape[1])
        image = tf.image.resize_with_crop_or_pad(image, side, side)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        targets = {
            "status": tf.one_hot(status, len(STATUS_NAMES)),
            "angle": angle,
        }
        weights = {"status": status_weight, "angle": angle_weight}
        return image, targets, weights

    return (
        dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_model():
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="appearance_augmentation",
    )
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    features = augmentation(inputs)
    features = base_model(features, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.3)(features)
    status = tf.keras.layers.Dense(len(STATUS_NAMES), activation="softmax", name="status")(features)
    angle_vector = tf.keras.layers.Dense(2, name="angle_components")(features)
    angle = tf.keras.layers.UnitNormalization(axis=-1, name="angle")(angle_vector)
    return tf.keras.Model(inputs, {"status": status, "angle": angle}), base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={"status": "categorical_crossentropy", "angle": "cosine_similarity"},
        loss_weights={"status": 1.0, "angle": 0.5},
        weighted_metrics={"status": ["accuracy"]},
    )


def prediction_outputs(prediction):
    values = list(prediction.values()) if isinstance(prediction, dict) else list(prediction)
    status = next((value for value in values if np.asarray(value).shape[-1] == 3), None)
    angle = next((value for value in values if np.asarray(value).shape[-1] == 2), None)
    if status is None or angle is None:
        raise RuntimeError("Expected model outputs with 3 status values and 2 angle values")
    return np.asarray(status), np.asarray(angle)


def evaluate_predictions(model, dataset):
    actual_status = []
    actual_angles = []
    target_masks = []
    for _, targets, weights in dataset:
        actual_status.append(np.argmax(targets["status"].numpy(), axis=1))
        actual_angles.append(targets["angle"].numpy())
        target_masks.append(weights["angle"].numpy().astype(bool))
    actual_status = np.concatenate(actual_status)
    actual_angles = np.concatenate(actual_angles)
    target_masks = np.concatenate(target_masks)
    status_probs, angle_vectors = prediction_outputs(model.predict(dataset, verbose=0))

    status_accuracy = float(np.mean(np.argmax(status_probs, axis=1) == actual_status))
    angle_errors = []
    for actual, predicted in zip(actual_angles[target_masks], angle_vectors[target_masks]):
        angle_errors.append(
            circular_difference_degrees(vector_to_angle(actual), vector_to_angle(predicted))
        )
    angle_mae = float(np.mean(angle_errors)) if angle_errors else math.nan
    print(f"Validation status accuracy: {status_accuracy * 100:.2f}%")
    print(f"Validation target angle MAE: {angle_mae:.2f} degrees")
    return status_accuracy, angle_mae


def train_and_export(
    manifest_path,
    prepared_dir="prepared_dataset",
    tflite_path="webbing_orientation_model.tflite",
    epochs=20,
    warmup_epochs=8,
    batch_size=16,
    frames_per_second=3.0,
    validation_fraction=0.2,
    seed=42,
    max_frames_per_video=None,
    overwrite_prepared=False,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    index_path = extract_manifest_frames(
        manifest_path,
        prepared_dir,
        frames_per_second,
        validation_fraction,
        seed,
        max_frames_per_video,
        overwrite_prepared,
    )
    training_samples = read_frame_index(index_path, "training")
    validation_samples = read_frame_index(index_path, "validation")
    training_dataset = make_dataset(training_samples, batch_size, True, seed)
    validation_dataset = make_dataset(validation_samples, batch_size, False, seed)

    model, base_model = build_model()
    tflite_path = Path(tflite_path).resolve()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(tflite_path.with_suffix(".best.weights.h5"))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_status_accuracy", mode="max", save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_status_accuracy", mode="max", patience=5, restore_best_weights=True
        ),
    ]

    print("\n--- Phase 1: training output heads ---")
    compile_model(model, 1e-3)
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=warmup_epochs,
        callbacks=callbacks,
    )

    print("\n--- Phase 2: fine-tuning backbone ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    compile_model(model, 1e-5)
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    model.load_weights(checkpoint_path)
    status_accuracy, angle_mae = evaluate_predictions(model, validation_dataset)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    metadata_path = tflite_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input": {"shape": [1, IMAGE_SIZE, IMAGE_SIZE, 3], "dtype": "float32", "range": [0, 255]},
                "status_labels": list(STATUS_NAMES),
                "angle_encoding": ["sin", "cos"],
                "validation": {"status_accuracy": status_accuracy, "angle_mae_degrees": angle_mae},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported model: {tflite_path}")
    print(f"Exported metadata: {metadata_path}")


def center_crop_and_resize(image):
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    crop = image[top : top + side, left : left + side]
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))


def run_inference(image_path, tflite_path="webbing_orientation_model.tflite"):
    image = np.asarray(Image.open(image_path).convert("RGB"))
    input_data = np.expand_dims(center_crop_and_resize(image), axis=0).astype(np.float32)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], input_data)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(detail["index"])[0] for detail in interpreter.get_output_details()]
    status_probs, angle_vector = prediction_outputs(outputs)
    status_probs = status_probs[0] if status_probs.ndim == 2 else status_probs
    angle_vector = angle_vector[0] if angle_vector.ndim == 2 else angle_vector
    status_index = int(np.argmax(status_probs))
    status = STATUS_NAMES[status_index]
    result = {"status": status, "confidence": float(status_probs[status_index])}
    if status == "target":
        result["angle_deg"] = vector_to_angle(angle_vector)
    print(json.dumps(result, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Train and run the multi-task webbing model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Extract labelled frames from manifest videos")
    train = subparsers.add_parser("train", help="Prepare videos, train, validate, and export TFLite")
    for command in (prepare, train):
        command.add_argument("--manifest", required=True, help="CSV with video,status,angle_deg columns")
        command.add_argument("--prepared-dir", default="prepared_dataset")
        command.add_argument("--fps", type=float, default=3.0, help="Frames sampled per second")
        command.add_argument("--validation-fraction", type=float, default=0.2)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--max-frames-per-video", type=int)
        command.add_argument("--overwrite-prepared", action="store_true")

    train.add_argument("--tflite", default="webbing_orientation_model.tflite")
    train.add_argument("--epochs", type=int, default=20, help="Fine-tuning epoch limit")
    train.add_argument("--warmup-epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=16)

    infer = subparsers.add_parser("infer", help="Run TFLite inference on one image")
    infer.add_argument("--image", required=True)
    infer.add_argument("--tflite", default="webbing_orientation_model.tflite")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "infer":
        run_inference(args.image, args.tflite)
        return
    if args.command == "prepare":
        extract_manifest_frames(
            args.manifest,
            args.prepared_dir,
            args.fps,
            args.validation_fraction,
            args.seed,
            args.max_frames_per_video,
            args.overwrite_prepared,
        )
        return
    train_and_export(
        args.manifest,
        args.prepared_dir,
        args.tflite,
        args.epochs,
        args.warmup_epochs,
        args.batch_size,
        args.fps,
        args.validation_fraction,
        args.seed,
        args.max_frames_per_video,
        args.overwrite_prepared,
    )


if __name__ == "__main__":
    main()
