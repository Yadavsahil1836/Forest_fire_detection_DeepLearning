# Forest Fire Detection Using Deep Learning

## Project Overview
This project implements a **Forest Fire Detection System** using **Deep Learning** models in **Google Colab**. It is part of the **Gen AI - AICTE Program**, aiming to leverage machine learning techniques for effective environmental monitoring.

### Objective
The goal of this project is to detect forest fires using a deep learning model trained on satellite images or any other dataset that contains visual or thermal data of forest fires. The system is expected to classify images into two categories: `Fire` and `No Fire`.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can clone this repository to your local machine and run the code in **Google Colab** to ensure all necessary packages and dependencies are installed.

```bash
git clone https://github.com/yadavsahil1836/Forest_fire_detection_DeepLearning.git
```
In Google Colab, simply upload the notebook and run the cells.

## Required Libraries
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

You can install all required libraries by running the following in a Colab notebook cell:

```bash
!pip install tensorflow keras numpy pandas opencv-python matplotlib scikit-learn
```

## Usage

1 . **Load the Dataset**: The first step is to load the dataset of images used for training the model. The dataset can consist of images of both forest fires and non-fire conditions.
2 . Preprocessing the Data: Image resizing, normalization, and augmentation are performed to prepare the data for the model.
3 . **Model Training**: The deep learning model is trained using a convolutional neural network (CNN), a popular model for image classification tasks.
4 . **Evaluation**: The trained model is evaluated on test data to check its performance and accuracy.

## Training the Model

To start training the model, run the following cell:

```bash
# Example: Training cell
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
```

## Inference

Once the model is trained, you can use it to predict fire conditions in new images:
```bash
# Example: Inference cell
predictions = model.predict(new_images)
```

## Dataset
The dataset used in this project is a collection of satellite images or images captured from various sources that include fire and non-fire labels. These datasets are publicly available or can be collected through various platforms.

## Model Architecture
The model implemented in this project uses a Convolutional Neural Network (CNN) to classify images into `Fire` or `No Fire` categories. The architecture consists of several layers, including convolutional layers, pooling layers, and fully connected layers. The model is trained using a binary cross-entropy loss function.

```bash
# Example model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])
```

## Results
The model’s performance can be evaluated using metrics like accuracy, precision, recall, and F1-score. A plot of the loss and accuracy curves during training and validation can be displayed to visualize the model's performance.

## Example Results
1. **Accuracy** : 92%
2. **Precision**: 91%
3. **Recall**   : 93%

# Contributing
If you'd like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Please ensure you follow the code of conduct and maintain consistency in the code.


