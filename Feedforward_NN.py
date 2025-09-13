#   Deep Learning Assignment  -|
#   ---------------------------|
#   Feedforward Neural Network (NN)  -|
#   ----------------------------------|


#   Introduction  -|
#   ---------------|
#   This assignment on Deep Learning focuses on implementing and analyzing neural network architectures for solving
#   real-world classification problems using standard datasets.
#   The project workflow includes:
#   - Feedforward Neural Network (NN): Building a multi-layer dense network to classify handwritten digits from the MNIST
#     dataset, using normalization, categorical cross-entropy loss, and the Adam optimizer.
#   - Convolutional Neural Network (CNN): Designing convolutional layers, pooling, dropout, and dense layers to classify
#     images from the CIFAR-10 dataset, with visualization of feature maps.
#   - Recurrent Neural Network (RNN/LSTM): Creating an LSTM-based model for sentiment analysis on IMDB movie reviews,
#     utilizing sequence padding, embedding layers, and binary classification.
#   - Data Preprocessing: Normalizing pixel values, reshaping images, tokenizing text, and encoding labels for model
#     readiness.
#   - Model Evaluation and Visualization: Training models, plotting loss and accuracy curves, and interpreting layer
#     activations.
#   The goal is to apply deep learning techniques, understand architectural differences between models, and gain
#   insights into how neural networks learn from structured and unstructured data for classification tasks.


import numpy as np
import tensorflow as tf
            # deep learning framework for creating and training machine learning models
from tensorflow.keras.datasets import mnist
            # importing data
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
            # modules to build neural networks
import matplotlib.pyplot as plt
            # for plotting graphs


tf.random.set_seed(42)
np.random.seed(42)
            # for reproducible results

(x_train, y_train), (x_test, y_test) = mnist.load_data()
            # loading data

x_train = x_train / 255.0
x_test = x_test / 255.0
            # scaling pixel values

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
            # converting integer labels into a vector of length 10

x_train = x_train.reshape(-1, 28 , 28)
x_test = x_test.reshape(-1, 28 , 28)

ffn_model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
            # creing a feedforward model while flattening to 1-D
            # with 2 hidden and 1 output layer

ffn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            # model configuration with optimiser, loss function and accuracy metrics

ffn = ffn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
            # model training

ffn_model.summary()
            # model architecture summary

test_loss, test_acc = ffn_model.evaluate(x_test, y_test)
            # model evaluation
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


plt.figure(figsize=(8,5))
plt.plot(ffn.history['accuracy'], label='Train Accuracy')
plt.plot(ffn.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
            # creating training and validation accuracy plots

plt.figure(figsize=(8,5))
plt.plot(ffn.history['loss'], label='Train Loss')
plt.plot(ffn.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
            # training and validation loss evolution plotting

plt.show()