
# Pneumonia-XRay-CNN-PyTorch — Chest X-ray Pneumonia Classification

## 📘 Description

This project implements a Convolutional Neural Network (CNN) using the `PyTorch` library to classify chest X-ray images as either `PNEUMONIA` or `NORMAL`. The objective is to build a robust model capable of assisting in medical diagnosis from images. The notebook provides a complete guide through the machine learning pipeline for a medical image classification problem.

The project focuses on key concepts such as data preprocessing, handling a significant class imbalance (many more pneumonia images than normal cases), and defining a CNN architecture using transfer learning for efficient and accurate training.

## 🛠️ Requirements

Ensure you have Python 3.8+ and the following libraries installed:

```bash
!pip install torch torchvision
!pip install kagglehub
!pip install pandas
```

Libraries used:

  * `torch`, `torch.nn`, `torch.optim` – Core PyTorch components for building and training the neural network.
  * `torchvision` – For data transformations and loading the image dataset.
  * `pandas` – For data manipulation.
  * `matplotlib.pyplot` – For data visualization and monitoring model performance.
  * `kagglehub` – For downloading the `chest-xray-pneumonia` dataset.

## 📊 Data Preprocessing

Data preprocessing is crucial, as the X-ray dataset has a significant class imbalance, with many more pneumonia images than normal cases. The notebook addresses this to prevent the model from becoming biased towards the majority class:

1.  **Dataset Download and Structure**: The `chest-xray-pneumonia` dataset is downloaded from Kaggle Hub. The images are organized into training, testing, and validation folders.
2.  **Visualization of Imbalance**: The number of images in each class (`NORMAL` and `PNEUMONIA`) for the training set is counted and displayed, confirming the imbalance.
3.  **Transformations**: A `transforms.Compose` pipeline is used to:
      * **Resize**: All images are resized to `224x224` pixels to standardize the model's input.
      * **Convert to Tensor**: Images are converted to PyTorch tensors, and pixel values are scaled to a range of `[0, 1]`.
      * **Normalize**: Images are normalized using the mean and standard deviation values for the RGB channel. This is a common practice for pre-trained models and accelerates convergence.

## 🧠 AI Model Structure

The model architecture is a CNN that leverages a pre-trained model to take advantage of existing feature learning.

  * **Transfer Learning**: A **ResNet-18** model pre-trained on the ImageNet dataset is used. Using transfer learning allows the model to leverage patterns and features it has already learned from a large dataset, and only needs to be fine-tuned for the specific pneumonia classification task. This is highly effective for moderately sized datasets.
  * **Freezing Layers**: The weights of most of ResNet-18's pre-trained layers are frozen to prevent them from being updated during training, thus preserving the general features they have already learned.
  * **Output Layer**: The final classification layer is modified to suit the specific problem: a `nn.Linear` layer that takes ResNet-18's output and projects it to 2 neurons (one for `NORMAL` and one for `PNEUMONIA`), followed by a `LogSoftmax` activation function.

## 🧬 Training and Results

The training process focuses on fine-tuning the model's classification layer.

  * **Training**: The model is trained for 5 epochs, using the `Adam` optimizer with a learning rate of 0.001.
  * **Loss and Accuracy**: `nn.CrossEntropyLoss` is used to measure the classification error. The model's loss and accuracy are calculated and printed for each epoch, allowing for performance monitoring on both the training and validation sets.
  * **Results**: The notebook demonstrates how the model achieves high accuracy in classifying chest X-ray images, showcasing the effectiveness of using transfer learning for this type of problem.

## 🚀 How to Run

1.  Download or clone the repository.
2.  Open the notebook:
    ```bash
    jupyter notebook L3P2-Pneumonia.ipynb
    ```
3.  Run the cells sequentially to:
      * Install the necessary libraries.
      * Download the dataset.
      * Preprocess the data and visualize the class imbalance.
      * Define and load the model architecture (`ResNet-18`).
      * Train the model.
      * Evaluate the results on the test and validation datasets.
