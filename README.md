# Deep Learning Assignment

## 📘 Overview

This repository contains the implementation of deep learning models for solving classification problems on standard datasets. The assignment covers three key architectures:

1. **Feedforward Neural Network (NN)** – Used for handwritten digit classification on the MNIST dataset.
2. **Convolutional Neural Network (CNN)** – Used for image classification on the CIFAR-10 dataset, with visualization of feature maps.
3. **Recurrent Neural Network (RNN/LSTM)** – Used for sentiment analysis on the IMDB movie reviews dataset.

The goal of this project is to apply deep learning techniques, understand various architectures, and explore how models perform on structured and unstructured data through training, evaluation, and visualization.

---

## 📂 Project Structure

- `Feedforward Neural Network (NN)`  
  Implements a dense network with two hidden layers for MNIST classification. Data preprocessing includes normalization and one-hot encoding. The model is evaluated using accuracy and loss plots.

- `Convolutional Neural Network (CNN)`  
  Implements convolutional layers, pooling, and dropout for CIFAR-10 classification. The model includes feature map visualization to understand how convolutional layers extract patterns from images.

- `Recurrent Neural Network (RNN/LSTM)`  
  Implements an LSTM model for text classification on IMDB reviews. Sequence padding and embedding layers are used, and training is monitored using accuracy and loss plots.

---

## 📥 Dataset Information

- **MNIST Dataset** – Handwritten digits, available from `tensorflow.keras.datasets`.
- **CIFAR-10 Dataset** – 60,000 32x32 color images in 10 classes, available from `tensorflow.keras.datasets`.
- **IMDB Dataset** – Movie reviews for sentiment analysis, available from `tensorflow.keras.datasets`.

---

## 🔢 Key Features

- Data preprocessing:
  - Normalizing pixel values.
  - One-hot encoding categorical labels.
  - Padding text sequences.

- Model architectures:
  - Dense layers with ReLU and Softmax activations.
  - Convolutional layers with pooling and dropout.
  - LSTM layers for sequence data.

- Evaluation and visualization:
  - Plotting training and validation accuracy and loss.
  - Model summaries for architecture inspection.
