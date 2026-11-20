# Brain Tumor MRI Classification

A Google Colab notebook workflow for loading, preparing, and inspecting the Brain Tumor MRI Dataset (`masoudnickparvar/brain-tumor-mri-dataset`) with TensorFlow/Keras image generators.

## Overview

This notebook sets up the data pipeline and visualization blocks for multi-class brain tumor classification across four categories:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Dataset

The workflow downloads and reads `masoudnickparvar/brain-tumor-mri-dataset` via `kagglehub`, expecting `Training` and `Testing` directories under `/kaggle/input/brain-tumor-mri-dataset/`:

- **Training set (`tr_df`)**: 5,600 labeled image paths loaded from `/kaggle/input/brain-tumor-mri-dataset/Training`.
- **Validation set (`valid_df`)**: 800 samples split (50% train size, `random_state=20`, stratified by `Class`) from the initial 1,600 testing records.
- **Test set (`ts_df`)**: The remaining 800 stratified samples after the validation split.

## Preprocessing and Data Generators

Images are resized to `(299, 299)` and passed through Keras `ImageDataGenerator` pipelines:

- **Training and Validation (`_gen`)**: Rescaled by `1/255` with `brightness_range=(0.8, 1.2)` and a batch size of `32`.
- **Testing (`ts_gen`)**: Rescaled by `1/255` with a batch size of `16` and `shuffle=False` for ordered evaluation and sample visualization.

## Dependencies

- `kagglehub`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `Pillow` (`PIL`)
- `scikit-learn`
- `tensorflow` (including Keras `Sequential`, `Dense`, `Dropout`, `Flatten`, `Adamax`, `Precision`, `Recall`, and `ImageDataGenerator`)

## Usage

1. Run the `kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset')` cell to fetch the dataset.
2. Import the required data science, visualization, `scikit-learn`, and TensorFlow/Keras modules.
3. Build `tr_df` and `ts_df` using `train_df` and `test_df`, plot class counts, and split `ts_df` into `valid_df` and `ts_df`.
4. Instantiate `tr_gen`, `valid_gen`, and `ts_gen`, preview sample images from `ts_gen`, and attach your `Sequential` model definition and training/evaluation cells.

## Notes

- The notebook includes a GPU accelerator setting (`T4`) in its Colab metadata and suppresses non-critical warnings via `warnings.filterwarnings('ignore')`.
- Model architecture compilation, training (`.fit`), and test evaluation (`classification_report`, `confusion_matrix`) are imported in the setup cell and ready to be added after the generator and preview steps.
