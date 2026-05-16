import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import display

import tensorflow as tf
from tensorflow import keras
import time as time


def plot_history(history, title="Träningskurvor"):
    """
    Plottar tränings- och valideringsförlust och noggrannhet över epoker.
    """

    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(12, 5))

    # Förlust
    plt.subplot(1, 2, 1)
    plt.plot(history_df["loss"], label="Training loss")
    plt.plot(history_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Noggrannhet
    plt.subplot(1, 2, 2)
    plt.plot(history_df["accuracy"], label="Training accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def model_compile_and_fit(model, X_train, y_train, X_val, y_val, name="Modell", epochs=20, optimizer="adam", learning_rate=0.001, callbacks=None):
    """
    Kompilerar och tränar modellen med given data och antal epoker.
    Returnerar träningshistoriken.
    """
    tf.keras.utils.set_random_seed(42)
    model = keras.models.clone_model(model)

    opt = keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_time = time.time()

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    if callbacks is None:
        callbacks = [early_stop]
    else:
        callbacks = callbacks + [early_stop]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        batch_size=32
    )

    training_time = time.time() - start_time

    print(f"\nResultat för {name}:")
    print(f"Antal epoker:        {len(history.history['loss'])}")
    print(f"Train accuracy:      {history.history['accuracy'][-1]:.4f}")
    print(f"Train loss:          {history.history['loss'][-1]:.4f}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Validation loss:     {history.history['val_loss'][-1]:.4f}")
    print(f"Träningstid:         {training_time:.2f} sekunder")

    plot_history(history, title=name)

    return model, history, {
        "name": name,
        "train_accuracy": history.history["accuracy"][-1],
        "train_loss": history.history["loss"][-1],
        "validation_accuracy": history.history["val_accuracy"][-1],
        "validation_loss": history.history["val_loss"][-1],
        "epochs_trained": len(history.history["loss"]),
        "training_time": training_time
    }