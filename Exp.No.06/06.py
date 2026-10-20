import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = (
    keras.datasets.fashion_mnist.load_data()
)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

def build_model():
    return keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

optimizers = {
    "SGD": keras.optimizers.SGD(learning_rate=0.01),
    "Momentum": keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9
    ),
    "Adam": keras.optimizers.Adam(learning_rate=0.001)
}

histories = {}

for name, optimizer in optimizers.items():
    tf.keras.utils.set_random_seed(7)
    model = build_model()
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    histories[name] = history.history
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(name, "Test loss:", round(loss, 4),
          "Test accuracy:", round(accuracy, 4))

for name, history in histories.items():
    plt.plot(history["val_loss"], marker="o", label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("SGD vs Momentum vs Adam")
plt.legend()
plt.show()