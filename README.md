# UW Madison GI Tract Image Segmentation Pipeline

This project provides a configurable pipeline using MONAI for the [UW-Madison GI Tract Image Segmentation Kaggle competition](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation). The goal is to automatically segment the stomach, small bowel, and large bowel in 3D abdominal MRI scans derived from 2D slices.

## Features

* Built upon the MONAI framework for medical image analysis.
* Configurable pipeline controlled via a central YAML file (`config.yaml`).
* Modular design with separate scripts for preprocessing, training, and inference.
* Handles conversion from 2D Kaggle data (PNG slices, RLE masks) to 3D NIfTI volumes.
* Supports standard data augmentation techniques for segmentation.
* Includes training loop with validation, metric logging (TensorBoard), and checkpointing.
* Basic inference script for generating predictions.

## File Structure & Roles

Here's a breakdown of the key files in this pipeline:

* **`config.yaml`:** Central configuration file defining all pipeline parameters and settings.
* **`preprocess_data.py`:** Converts raw 2D PNGs and RLE masks into 3D NIfTI volumes for analysis (saves image and separate mask channels).
* **`utils.py`:** Provides shared helper functions for configuration loading, RLE handling, path parsing, and other utilities.
* **`transforms.py`:** Defines MONAI transform sequences for data loading, augmentation, mask channel concatenation, and processing based on `config.yaml`.
* **`dataset.py`:** Creates MONAI Datasets and DataLoaders to feed processed data batches to the model.
* **`model_factory.py`:** Instantiates the specified neural network model architecture based on `config.yaml`.
* **`losses.py`:** Defines and provides the loss function used during model training based on `config.yaml`.
* **`metrics.py`:** Defines and provides the evaluation metrics used during validation based on `config.yaml`.
* **`train.py`:** Orchestrates the end-to-end model training loop, including validation and checkpointing.
* **`inference.py`:** Loads a trained model to generate predictions on new data and formats the output.
* **`requirements.txt`:** Lists the required Python libraries needed to set up the project environment.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/sakura657/SegPipeline.git
    cd SegPipeline
    ```
2.  **Create Environment & Install Dependencies:**
    ```bash
    conda create -n SegPip python=3.10 -y
    conda activate SegPip
    pip install -r requirements.txt
    ```
3.  **Download Data:**
    * Download the competition data from [Kaggle: UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data).
    * You primarily need the `train/` folder (containing `caseX/caseX_dayY/scans/*.png` files) and the `train.csv` file.
4.  **Place Data:**
    * Arrange the downloaded data so that the path specified in `config.yaml` under `data.raw_data_dir` points to the directory containing `train.csv` and the `train` folder.
    * Example: If `config.yaml` has `data.raw_data_dir: "../uw-madison-gi-tract-image-segmentation"`, place the Kaggle data in a folder named `uw-madison-gi-tract-image-segmentation` at the same level as your `SegPipeline` project folder.

## Usage

The main workflow involves preprocessing, training, and inference, primarily controlled via the configuration file.

1.  **Configure Pipeline:**
    * Modify `config.yaml` to set data paths, model parameters (architecture, channels, etc.), training hyperparameters (learning rate, batch size, epochs), augmentations, loss function, fold number, etc.

2.  **Preprocess Data:**
    * Run the preprocessing script. This converts the raw 2D data into 3D NIfTI files (`_image.nii.gz`, `_mask_lb.nii.gz`, `_mask_sb.nii.gz`, `_mask_st.nii.gz`) and creates JSON files for data splits. This step is typically run once per configuration.
    ```bash
    python preprocess_data.py -c config.yaml
    ```
    * *(Note: By default, this script processes data for all 5 folds. You can process a specific fold using `-f <fold_number>`)*

3.  **Train Model:**
    * Start the training process using the specified configuration. Ensure the `fold` parameter in `config.yaml` matches the preprocessed data you want to use.
    ```bash
    python train.py -c config.yaml
    ```
    * *(You can override the fold number in the config file for a specific run using `-f <fold_number>`)*
    * Logs and checkpoints will be saved to the directories specified in `config.yaml` (e.g., `./output/fold_X/`).

4.  **Run Inference:**
    * Generate predictions using a trained model. Make sure the `inference.checkpoint_path` in `config.yaml` points to the desired model checkpoint (`.pth` file).
    ```bash
    python inference.py -c config.yaml
    ```
    * *(You can override the checkpoint path in the config file for a specific run using `--ckpt <path_to_checkpoint.pth>`)*
    * This will typically generate a `submission.csv` file in the inference output directory.

## Configuration

Most pipeline behavior is controlled through `config.yaml`. Refer to the comments within the file for details on specific parameters. Key sections include `data`, `preprocessing_3d`, `augmentation`, `model`, `training`, `inference`, `postprocessing`, and `evaluation`.

## Dependencies

See `requirements.txt`. Major dependencies include:

* PyTorch
* MONAI
* SimpleITK
* Pillow
* NumPy
* Pandas
* scikit-image
* PyYAML
* python-box