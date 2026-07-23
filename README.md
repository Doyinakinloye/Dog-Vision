# 🐶 Dog Vision — End-to-End Multi-Class Dog Breed Classifier

An end-to-end deep learning image classifier that identifies a dog's breed from a photo, built with TensorFlow and TensorFlow Hub using transfer learning on **MobileNetV2**.

## Problem

Given a photo of a dog, predict which of 120 breeds it belongs to — e.g. spot a dog at a cafe and instantly know its breed.

## Data

- Source: [Kaggle Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification/data)
- **120** unique dog breeds
- **10,222** labeled training images, **10,357** unlabeled test images
- Labels provided as a CSV mapping image ID → breed name

## Evaluation

Kaggle scores submissions on a CSV of prediction probabilities for every dog breed, for every test image (multi-class log loss). The notebook produces a submission file in this exact format.

## Approach

1. **Load & explore the data** — read `labels.csv`, inspect class balance (breed counts range around a median of ~82 images per breed) and sample images.
2. **Prepare inputs/outputs**
   - Build the list of image filepaths from image IDs.
   - One-hot / boolean-encode the 120 breed labels.
   - Split off a custom validation set (Kaggle doesn't provide one) using `train_test_split`.
3. **Turn images into tensors**
   - Read each JPEG, decode it, normalize pixel values to `[0, 1]`, and resize to `224 × 224`.
4. **Batch the data** — build `tf.data.Dataset` pipelines that shuffle (training only) and batch images/labels into `(image, label)` tuples (batch size 32) for memory-efficient training.
5. **Build the model** — transfer learning with **MobileNetV2** (ImageNet weights, top layer removed, frozen base) + `GlobalAveragePooling2D` + a `Dense(120, activation="softmax")` output layer.
6. **Callbacks** — TensorBoard (for tracking experiments) and `EarlyStopping` (monitoring accuracy/validation accuracy) to prevent overfitting.
7. **Train in two stages**
   - First on a smaller subset of the training data to validate the full pipeline quickly.
   - Then on the **full training set** once the pipeline is confirmed to work.
8. **Evaluate & visualize predictions** — plot predicted vs. true labels on sample images, plus top-10 prediction confidence bar charts.
9. **Save / reload the model** (`.h5` format, with the custom `KerasLayer` object) so training doesn't need to be repeated.
10. **Predict on the test set** and format the output as a Kaggle-ready submission CSV (`id` column + one probability column per breed).

## Results

| Stage | Data | Result |
|---|---|---|
| Prototype model | Subset (800 train / 200 val images) | ~62.5% validation accuracy |
| Full model | All 10,222 training images | >99% training accuracy after ~8 epochs (no held-out validation set used for the final full-data run) |

The prototype run on a small subset was used to sanity-check the pipeline before committing to a longer, full-dataset training run.

## Tech Stack

- Python, Google Colab (GPU runtime)
- TensorFlow / Keras
- TensorFlow Hub (`hub.KerasLayer`)
- MobileNetV2 (pretrained on ImageNet, via `tf.keras.applications`)
- pandas / NumPy for data handling
- Matplotlib for visualization
- scikit-learn (`train_test_split`)
- TensorBoard for experiment tracking

## Getting Started

This project was built and run in **Google Colab** with data stored on Google Drive.

1. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/dog-breed-identification/data) and place it in your Google Drive (e.g. `drive/MyDrive/Dog Vision/`), with `train/`, `test/`, and `labels.csv`.
2. Open `Dog_Vision.ipynb` in Colab and mount your Google Drive.
3. Run all cells top to bottom. The notebook will:
   - Load and explore the labeled data
   - Preprocess images into batched Tensors
   - Train a MobileNetV2-based transfer learning model (first on a subset, then on the full dataset)
   - Evaluate and visualize predictions
   - Generate `Predictions.csv` in the Kaggle submission format

### Requirements

```
tensorflow
tensorflow_hub
pandas
numpy
scikit-learn
matplotlib
```

## Project Structure Notes

- `process_image()` — reads, decodes, normalizes, and resizes a single image to `224 × 224 × 3`.
- `create_data_batches()` — builds shuffled/batched `tf.data.Dataset` pipelines for train, validation, or test data.
- `create_model()` — builds the MobileNetV2 transfer-learning model.
- `train_model()` — trains a fresh model with TensorBoard + early stopping callbacks.
- `save_model()` / `load_model()` — persist and restore trained models.
- `get_pred_label()`, `plot_pred()`, `plot_pred_conf()` — utilities for turning prediction probabilities into human-readable breed labels and visualizations.

## Future Work

- Fine-tune the base MobileNetV2 layers (rather than keeping them fully frozen) for potentially higher accuracy.
- Add data augmentation to improve generalization.
- Experiment with other pretrained backbones (e.g. EfficientNet, ResNet) for comparison.
- Track a proper held-out validation split even during full-dataset training to better monitor generalization.

## Acknowledgements

- [Kaggle Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification) for the dataset and evaluation format
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — pretrained ImageNet weights via `tf.keras.applications`
