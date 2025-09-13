#   Recurrent Neural Network (RNN/LSTM)  -|
#   --------------------------------------|


import numpy as np
import tensorflow as tf
            # deep learning framework for creating and training machine learning models
from tensorflow.keras.datasets import imdb
            # importing data
from tensorflow.keras.models import models, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
            # modules to build neural networks
import matplotlib.pyplot as plt
            # for plotting graphs


tf.random.set_seed(42)
np.random.seed(42)
            # for reproducible results

words_limit = 10000
            # vocabulary size
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=words_limit)
            # loading data

maxlen = 200
            # maximum sequence length to 200 tokens
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
            # padding sequences to ensure uniform input size
            # sequences longer than maxlen are truncated and shorter ones are padded with zeros

LSTM_model = models.Sequential([
    layers.Embedding(input_dim=words_limit, output_dim=32, input_length=maxlen),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
            # creating an LSTM model with embedding layer, LSTM layer
            # dropout layer and output layer

LSTM_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
            # model configuration with optimiser, loss function and accuracy metrics

LSTM_model.summary()
            # model architecture summary

LSTM = LSTM_model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
            # model training

test_loss, test_acc = LSTM_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
            # model evaluation

# Plot training and validation metrics
plt.figure(figsize=(8,5))
plt.plot(LSTM.history['accuracy'], label='Train Accuracy')
plt.plot(LSTM.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
            # creating training and validation accuracy plots

plt.figure(figsize=(8,5))
plt.plot(LSTM.history['loss'], label='Train Loss')
plt.plot(LSTM.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
            # training and validation loss evolution plotting




#   Conclusion  -|
#   -------------|
#   The assignment successfully implemented deep learning pipelines for image and text classification using feedforward,
#   convolutional, and recurrent neural networks.
#   Key Findings:
#   - The feedforward network effectively classified handwritten digits, demonstrating the impact of activation functions and
#     optimizer choices on model performance.
#   - The CNN architecture significantly improved image classification by capturing spatial patterns through convolution and
#     pooling operations, with dropout layers helping prevent overfitting.
#   - The LSTM model showcased how sequential data, like movie reviews, can be processed for sentiment classification by
#     learning temporal dependencies.
#   - Visualization of training and validation metrics provided insights into convergence, learning rates, and potential
#     areas of model refinement.
#   - Feature map visualization illustrated how convolutional layers extract hierarchical patterns from images.
#   Overall, the assignment highlighted the power and versatility of deep learning models in solving complex classification
#   problems, reinforcing best practices in data preprocessing, model building, evaluation, and interpretation for research
#   and practical applications.