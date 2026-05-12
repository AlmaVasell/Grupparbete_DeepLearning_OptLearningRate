import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras


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

def model_compile_and_fit(model, X_train, y_train, X_val, y_val, epochs=20, optimizer="adam", learning_rate=0.001):
    """
    Kompilerar och tränar modellen med given data och antal epoker.
    Returnerar träningshistoriken.
    """

    opt = keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        batch_size=32
    )

    return history