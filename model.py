"""
model.py — Phase 3: CNN Architecture for spectrogram-based stock prediction.

Architecture:
  Input (F × T_frames × C)
    → Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    → Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    → Conv2D(128, 3×3) + BatchNorm + ReLU
    → GlobalAveragePooling2D
    → Dense(128) + Dropout(0.3)
    → Dense(64) + ReLU
    → Dense(1)   ← predicted normalized Close price
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape):
    """
    Build the CNN regression model.

    Parameters
    ----------
    input_shape : tuple — X_train.shape[1:], e.g. (17, 5, 17)
                  = (freq_bins, time_frames, n_channels)

    Returns
    -------
    tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    # ── Block 1 ──────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ── Block 2 ──────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ── Block 3 ──────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ── Regression Head ──────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1)(x)            # single output: predicted price

    return models.Model(inputs, output, name="StockCNN")


def compile_model(model, lr=0.001):
    """Compile with Adam optimizer and MSE loss (assignment requirement)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_cnn((17, 5, 17))
    m = compile_model(m)
    m.summary()
