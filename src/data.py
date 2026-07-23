"""Dataset discovery and reproducible stratified splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetSplit:
    """Paths and labels for train, validation, and test partitions."""

    train_paths: np.ndarray
    train_labels: np.ndarray
    validation_paths: np.ndarray
    validation_labels: np.ndarray
    test_paths: np.ndarray
    test_labels: np.ndarray


def discover_labeled_images(
    dataset_dir: Path,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Discover image paths under one subdirectory per class."""

    paths: list[str] = []
    labels: list[int] = []

    for label, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        class_paths = sorted(
            str(path)
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(f"No supported images found in {class_dir}")

        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))

    return np.asarray(paths), np.asarray(labels, dtype=np.int64)


def stratified_split(
    paths: Sequence[str],
    labels: Sequence[int],
    seed: int = 42,
) -> DatasetSplit:
    """Create a reproducible stratified 80/10/10 split."""

    path_array = np.asarray(paths)
    label_array = np.asarray(labels)

    if len(path_array) != len(label_array):
        raise ValueError("paths and labels must have the same length")
    if len(np.unique(label_array)) < 2:
        raise ValueError("at least two classes are required")

    train_paths, temporary_paths, train_labels, temporary_labels = train_test_split(
        path_array,
        label_array,
        test_size=0.20,
        stratify=label_array,
        random_state=seed,
    )
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(
        temporary_paths,
        temporary_labels,
        test_size=0.50,
        stratify=temporary_labels,
        random_state=seed,
    )

    return DatasetSplit(
        train_paths=train_paths,
        train_labels=train_labels,
        validation_paths=validation_paths,
        validation_labels=validation_labels,
        test_paths=test_paths,
        test_labels=test_labels,
    )
