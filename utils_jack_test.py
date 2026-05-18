import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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



def model_compile_and_fit(model, X_train, y_train, X_val, y_val, name="Modell", epochs=50, optimizer="adam", learning_rate=0.01, callbacks=None, verbose_output=True, patience=3):
    """
    Kompilerar och tränar modellen med given data och antal epoker.
    Returnerar träningshistoriken.
    """

    model = keras.models.clone_model(model)

    opt = keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_time = time.time()

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

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

    # Ändrat så vi inte skriver ut värden från sista epok utan skriver ut bästa värdet
    training_time = time.time() - start_time

    best_index = np.argmax(history.history["val_accuracy"])

    best_train_accuracy = history.history["accuracy"][best_index]
    best_train_loss = history.history["loss"][best_index]
    best_val_accuracy = history.history["val_accuracy"][best_index]
    best_val_loss = history.history["val_loss"][best_index]


    if verbose_output:
        print(f"\nResultat för {name}:")
        print(f"Antal epoker:        {len(history.history['loss'])}")
        print(f"Train accuracy:      {best_train_accuracy:.4f}")
        print(f"Train loss:          {best_train_loss:.4f}")
        print(f"Validation accuracy: {best_val_accuracy:.4f}")
        print(f"Validation loss:     {best_val_loss:.4f}")
        print(f"Träningstid:         {training_time:.2f} sekunder")

        plot_history(history, title=name)


    return model, history, {
    "name": name,
    "train_accuracy": best_train_accuracy,
    "train_loss": best_train_loss,
    "validation_accuracy": best_val_accuracy,
    "validation_loss": best_val_loss,
    "epochs_trained": len(history.history["loss"]),
    "learning_rate": learning_rate,
    "training_time": training_time
    }