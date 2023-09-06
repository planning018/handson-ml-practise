import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import netural_util
from packaging import version

assert version.parse(tf.__version__) >= version.parse("2.8.0")
print("Tensorflow version: ", tf.__version__)

"""
Using Keras to load the dataset
"""
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

print("X_train.shape is: \n", X_train.shape)
print("X_train.dtype is: \n", X_train.dtype)
print("y_train is: \n", y_train)

# let's scale the pixel intensities down the 0-1 range
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

"""
Let's take a look at a sample of the images in the dataset
"""
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)

netural_util.save_fig("fashion_mnist_plot")
plt.show()

"""
Creating the model using the Sequential API
"""
tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

"""
This is equivalent to:
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
"""
model.summary()

# tf.keras.utils.plot_model(model, "my_fashion_mnist_model.jpg", show_shapes=True)
print("Model layers: ", model.layers)
hidden1 = model.layers[1]
print("hidden1 layer name is : ", hidden1.name)
print("compare layers result is : ", model.get_layer('dense') is hidden1)
weights, biases = hidden1.get_weights()
print("weights shape is : ", weights.shape)
print("weights is : ", weights)

"""
Compiling the model
"""
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

"""
This is equivalent to:
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
"""
# convert class ids to one-hot vectors
tf.keras.utils.to_categorical([0, 5, 1, 0], num_classes=10)
# extra code – shows how to convert one-hot vectors to class ids
np.argmax(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    axis=1
)

"""
Training and evaluating the model
"""
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
print("history params is : \n", history.params)
print("history epochs is: \n", history.epoch)


pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")  # extra code
netural_util.save_fig("keras_learning_curves_plot")  # extra code
plt.show()

# extra code – shows how to shift the training curve by -1/2 epoch
plt.figure(figsize=(8, 5))
for key, style in zip(history.history, ["r--", "r--.", "b-", "b-*"]):
    epochs = np.array(history.epoch) + (0 if key.startswith("val_") else -0.5)
    plt.plot(epochs, history.history[key], style, label=key)
plt.xlabel("Epoch")
plt.axis([-0.5, 29, 0., 1])
plt.legend(loc="lower left")
plt.grid()
plt.show()

model.evaluate(X_test, y_test)


"""
Using the model to make predictions
"""
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))
y_pred = y_proba.argmax(axis=-1)
print("y_pred is : ", y_pred)
print("y_pred class names: ", np.array(class_names)[y_pred])
print("y_new is: ", y_test[:3])


# extra code – this cell generates and saves Figure 10–12
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
netural_util.save_fig('fashion_mnist_images_plot', tight_layout=False)
plt.show()
