#   Convolutional Neural Network (CNN)  -|
#   -------------------------------------|


import numpy as np
import tensorflow as tf
            # deep learning framework for creating and training machine learning models
from tensorflow.keras.datasets import cifar10
            # importing data
from tensorflow.keras.models import  models, layers
from tensorflow.keras.utils import to_categorical
            # modules to build neural networks
import matplotlib.pyplot as plt
            # for plotting graphs


tf.random.set_seed(42)
np.random.seed(42)
            # for reproducible results

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
            # loading data

x_train = x_train / 255.0
x_test = x_test / 255.0
            # scaling pixel values

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
            # converting integer labels

inputs = x_train.shape[1:]
cnn_model =  models.Sequential([
    layers.Input(shape = inputs),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
            # creating a CNN model with 4 convolutional layers, 2 max-pooling layers

cnn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            # model configuration with optimiser, loss function and accuracy metrics


cnn = cnn_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)
            # model training

cnn_model.summary()
            # model architecture summary

# Evaluate
test_loss, test_acc = cnn_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
            # model evaluation

# Plot accuracy and loss
plt.figure(figsize=(8,5))
plt.plot(cnn.history['accuracy'], label='Train Accuracy')
plt.plot(cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
            # creating training and validation accuracy plots

plt.figure(figsize=(8,5))
plt.plot(cnn.history['loss'], label='Train Loss')
plt.plot(cnn.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
            # training and validation loss evolution plotting


cnn_model.build(input_shape=(None, 32, 32, 3))
            # model building with input specified shape

layer_outputs = [layer.output for layer in cnn_model.layers if isinstance(layer, Conv2D)]
            # generating list of output tensors for all convolutional layers in the cnn_model
activation_model = Model(inputs=cnn_model.input, outputs=layer_outputs)
            # creating a new model with same input as the cnn_model

img = x_test[0].reshape(1,32,32,3)
            # reshaping the first image from test set
activations = activation_model.predict(img)

first_layer_activation = activations[0]
plt.figure(figsize=(15,15))
for i in range(6):
    plt.subplot(1,6,i+1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
            # plot showing the activation pattern
plt.tight_layout()
            # avoiding overlap between subplots
plt.show()