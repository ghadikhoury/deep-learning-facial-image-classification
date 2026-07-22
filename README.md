# Facial Image Classification for Autism Screening Research

A comparative study of eight neural network architectures — four trained from scratch and four using transfer learning — for binary classification of facial images (autistic / non-autistic) using TensorFlow/Keras.

> **Research context, not a diagnostic tool.** This project is an academic exercise in comparative CNN architecture design and transfer learning, built on a public labeled image dataset. Facial-image-based autism classification is scientifically contested and this model is **not validated for, and should not be used for, real-world screening, diagnosis, or clinical decision-making.**

## Overview

The project compares two families of approaches on the same dataset and preprocessing pipeline:

1. **Custom architectures trained from scratch** — a baseline CNN, an enhanced CNN with dropout/L2 regularization, a deeper CNN, and a fully connected (dense) network — to establish how much architecture choice alone affects performance.
2. **Transfer learning** using ImageNet-pretrained **MobileNetV2** and **ResNet50**, each evaluated both frozen (feature extraction only) and fine-tuned, to measure the lift from pretrained visual features on a small, domain-specific dataset.

## Dataset

- 2,526 labeled facial images across 2 classes (`autistic`, `non_autistic`)
- Stratified split, fixed seed (`SEED=42`): **2,020 train / 253 validation / 253 test** (80/10/10)
- Images resized to 180×180, batch size 32
- On-the-fly augmentation: random horizontal/vertical flip, rotation, zoom, and contrast adjustment

## Models Compared

| # | Model | Approach | Key techniques |
|---|-------|----------|-----------------|
| 1 | Base CNN | From scratch | 4× Conv2D/MaxPool blocks |
| 2 | Enhanced CNN | From scratch | + L2 regularization, dropout (0.3) |
| 3 | Deep CNN | From scratch | Deeper conv stack, dropout (0.25–0.5) |
| 4 | DNN (fully connected) | From scratch | Dense-only baseline, L2 regularization |
| 5 | MobileNetV2 (frozen) | Transfer learning | ImageNet weights, frozen base |
| 6 | ResNet50 (frozen) | Transfer learning | ImageNet weights, frozen base |
| 7 | MobileNetV2 (fine-tuned) | Transfer learning | Last 20 layers unfrozen |
| 8 | ResNet50 (fine-tuned) | Transfer learning | Two-stage: head training → gradual unfreeze of final layers, LR scheduling |

All models trained with early stopping, model checkpointing, and (where applicable) learning-rate reduction on plateau.

## Results

Validation accuracy by best epoch, taken directly from training logs:

| Model | Best val. accuracy | Notes |
|---|---|---|
| Base CNN | — | Run cut short in original notebook session |
| Enhanced CNN (dropout + L2) | 71.9% | Stable, no major overfitting |
| Deep CNN | ~50% | Failed to converge past chance level |
| DNN (fully connected) | ~60% | Unstable / diverging loss — expected for a non-convolutional baseline on image data |
| MobileNetV2 (frozen) | 80.1% | |
| ResNet50 (frozen) | 80.9% | |
| MobileNetV2 (fine-tuned) | 82.4% | |
| **ResNet50 (fine-tuned, two-stage)** | **87.9% (val)** | |

**Final held-out test set — best model (fine-tuned ResNet50):**

| Metric | Value |
|---|---|
| Accuracy | **83.5%** |
| AUC | **0.917** |
| Loss | 0.400 |

The from-scratch Deep CNN and DNN baselines are included deliberately: they show that architecture depth alone doesn't help without either regularization or pretrained features, motivating the move to transfer learning.

## Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `NumPy` · `Matplotlib` · `PIL`

## Repository Structure

```
.
├── notebooks/
│   └── facial_autism_classification.ipynb   # Full training & evaluation pipeline
├── requirements.txt
└── README.md
```

## Running It

The notebook was developed in Google Colab and expects the dataset mounted from Google Drive. To run it:

1. `pip install -r requirements.txt`
2. Update `base_dir` in the notebook to point to your local copy of the dataset (structured as `<base_dir>/autistic/` and `<base_dir>/non_autistic/`)
3. Run cells top to bottom

## Limitations & Honest Notes

- The from-scratch Deep CNN and DNN models did not converge well on this dataset size — included here as a comparison point, not a strong result.
- Dataset size (2,526 images) is small by deep learning standards; results should be read as a controlled architecture comparison, not a claim of generalizable clinical performance.
- This is exploratory academic work, not a peer-reviewed or clinically validated method.
