# Pneumonia Detection from Chest X-Rays using Deep Learning

This repository contains a Convolutional Neural Network (CNN) pipeline developed to classify chest X-ray images into two classes: **Normal** and **Pneumonia**.

## Project Overview
Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus. Early detection through chest X-rays is vital. This project aims to build an automated, accurate, and robust Deep Learning system that assists medical professionals in diagnosing pneumonia.

The project demonstrates incremental model development—starting from a baseline Convolutional Neural Network and advancing to a complex, fine-tuned `ResNet50V2` architecture.

## Methodology

### 1. Data Preprocessing & Augmentation
*   **Data Resampling:** Images are reshaped, normalized, and mapped correctly to their classes.
*   **Balancing Classes:** Class weights are calculated specifically to combat class imbalance in medical datasets.
*   **Data Augmentation:** The training pipeline incorporates **RandomFlip**, **RandomRotation**, and **RandomZoom** layers to reduce overfitting and improve model generalization on limited data.

### 2. Modeling Iterations
1.  **Baseline CNN:** A starting model block with standard `Conv2D` layers, `MaxPooling2D`, and simple dense classification heads.
2.  **Deeper Network with Regularization:** Stacking more convolutional layers combined with Dropout blocks (`0.5`) to penalize heavy weights and avoid memorization.
3.  **Transfer Learning with ResNet50V2:** Integrating a pre-trained robust `ResNet50V2` model utilizing ImageNet weights. The process involved:
    *   Building an adapter to convert the initial generic image layers to the 3-channel input required by ResNet.
    *   **Feature Extraction:** Training the custom dense head while the base model is frozen.
    *   **Fine-Tuning:** Unfreezing the base model completely and training it with a highly constrained learning rate (`1e-5`).

### 3. Optimization Callbacks
The training leverages essential callbacks to maintain stable optimization:
*   `EarlyStopping` - Restores the best weights to halt training if validation loss stops improving.
*   `ReduceLROnPlateau` - Dynamically decays the learning rate to escape local minima.
*   `ModelCheckpoint` - Automatically saves the best model found during training.

## Usage
1.  Clone this repository or load `Pneumonia_Detection.ipynb` into a Jupyter or Google Colab environment.
2.  Install dependencies included in the `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3.  The final fine-tuned model weights are saved natively as `best_model.keras` which can be loaded as:
    ```python
    import tensorflow as tf
    best_model = tf.keras.models.load_model('best_model.keras')
    ```

## Dataset Statement
The notebook automatically downloads the [yusufmurtaza01/chest-xray-pneumonia-balanced-dataset](https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset/data) using the `kagglehub` API.
