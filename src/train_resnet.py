"""Train and evaluate a two-stage ResNet50 image classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.data import DatasetSplit, discover_labeled_images, stratified_split


CLASS_NAMES = ("autistic", "non_autistic")


@dataclass(frozen=True)
class TrainingConfig:
    """Training and data-pipeline configuration."""

    image_size: tuple[int, int] = (180, 180)
    batch_size: int = 32
    seed: int = 42
    head_epochs: int = 10
    fine_tune_epochs: int = 15
    fine_tune_layers: int = 20
    head_learning_rate: float = 3e-4
    fine_tune_learning_rate: float = 1e-5


def set_reproducible_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig,
    training: bool,
) -> tf.data.Dataset:
    """Build a decoded, resized, batched, and prefetched dataset."""

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, config.image_size, antialias=True)
        return image, tf.cast(label, tf.float32)

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_model(
    config: TrainingConfig,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Create a frozen ImageNet ResNet50 with a binary classifier head."""

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*config.image_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*config.image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs), base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def make_callbacks(output_dir: Path, stage: str) -> list[tf.keras.callbacks.Callback]:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
        ),
        ModelCheckpoint(
            output_dir / f"{stage}_best.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]


def build_datasets(
    split: DatasetSplit,
    config: TrainingConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    return (
        make_dataset(split.train_paths, split.train_labels, config, training=True),
        make_dataset(
            split.validation_paths,
            split.validation_labels,
            config,
            training=False,
        ),
        make_dataset(split.test_paths, split.test_labels, config, training=False),
    )


def train(
    dataset_dir: Path,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> dict[str, float | str]:
    """Train two stages, select by validation loss, and test once."""

    config = config or TrainingConfig()
    set_reproducible_seeds(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = discover_labeled_images(dataset_dir, CLASS_NAMES)
    split = stratified_split(paths, labels, seed=config.seed)
    train_dataset, validation_dataset, test_dataset = build_datasets(split, config)

    print(
        "Split sizes:",
        {
            "train": len(split.train_paths),
            "validation": len(split.validation_paths),
            "test": len(split.test_paths),
        },
    )

    model, base_model = build_model(config)
    compile_model(model, config.head_learning_rate)
    head_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.head_epochs,
        callbacks=make_callbacks(output_dir, "head"),
    )

    base_model.trainable = True
    for layer in base_model.layers[: -config.fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-config.fine_tune_layers :]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, config.fine_tune_learning_rate)
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.fine_tune_epochs,
        callbacks=make_callbacks(output_dir, "fine_tuned"),
    )

    head_best_loss = min(head_history.history["val_loss"])
    fine_tuned_best_loss = min(fine_tune_history.history["val_loss"])
    if fine_tuned_best_loss < head_best_loss:
        selected_stage = "fine_tuned"
    else:
        selected_stage = "head"

    model.load_weights(output_dir / f"{selected_stage}_best.weights.h5")
    test_results = model.evaluate(test_dataset, return_dict=True)

    metrics: dict[str, float | str] = {
        "selected_stage": selected_stage,
        "head_best_validation_loss": float(head_best_loss),
        "fine_tuned_best_validation_loss": float(fine_tuned_best_loss),
        "test_loss": float(test_results["loss"]),
        "test_accuracy": float(test_results["accuracy"]),
        "test_auc": float(test_results["auc"]),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a two-stage ResNet50 facial image classifier."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Directory containing autistic/ and non_autistic/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/resnet50"),
        help="Directory for checkpoints and metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.dataset, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
