import numpy as np
import pytest

from src.data import stratified_split


def make_samples(per_class: int = 100) -> tuple[np.ndarray, np.ndarray]:
    paths = np.array(
        [f"class_0/image_{index}.jpg" for index in range(per_class)]
        + [f"class_1/image_{index}.jpg" for index in range(per_class)]
    )
    labels = np.array([0] * per_class + [1] * per_class)
    return paths, labels


def test_stratified_split_sizes_and_no_path_overlap() -> None:
    paths, labels = make_samples()
    split = stratified_split(paths, labels, seed=42)

    assert len(split.train_paths) == 160
    assert len(split.validation_paths) == 20
    assert len(split.test_paths) == 20

    train = set(split.train_paths)
    validation = set(split.validation_paths)
    test = set(split.test_paths)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_stratification_preserves_class_balance() -> None:
    paths, labels = make_samples()
    split = stratified_split(paths, labels, seed=42)

    assert np.bincount(split.train_labels).tolist() == [80, 80]
    assert np.bincount(split.validation_labels).tolist() == [10, 10]
    assert np.bincount(split.test_labels).tolist() == [10, 10]


def test_split_is_reproducible() -> None:
    paths, labels = make_samples()
    first = stratified_split(paths, labels, seed=7)
    second = stratified_split(paths, labels, seed=7)

    assert np.array_equal(first.train_paths, second.train_paths)
    assert np.array_equal(first.validation_paths, second.validation_paths)
    assert np.array_equal(first.test_paths, second.test_paths)


def test_rejects_mismatched_paths_and_labels() -> None:
    with pytest.raises(ValueError, match="same length"):
        stratified_split(["a.jpg", "b.jpg"], [0])
