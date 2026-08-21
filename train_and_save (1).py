"""
train_and_save.py
Run once locally to train the MNIST model and save it.
Usage: python train_and_save.py
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import os

print("Training MNIST ANN …")

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test  = X_test  / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(60,  activation="relu"),
    Dense(10,  activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=4, batch_size=32, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

model.save("mnist_model.keras")
size = os.path.getsize("mnist_model.keras")
print(f"Saved → mnist_model.keras  ({size:,} bytes)")
