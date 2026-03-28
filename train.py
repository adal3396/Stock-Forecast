"""
train.py — Phase 3: Training Pipeline

Workflow:
  1. Download stock data for configured tickers
  2. Normalize all features
  3. Build STFT spectrogram dataset
  4. Train CNN with EarlyStopping + ReduceLROnPlateau
  5. Save model, history, scalers, and test data

Run:  python train.py
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from preprocess import (
    download_stock_data, normalize,
    make_dataset, train_val_test_split
)
from model import build_cnn, compile_model

# ── CONFIG ──────────────────────────────────────────────────────────
TICKERS         = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
START_DATE      = "2018-01-01"
END_DATE        = "2024-01-01"
TARGET_COL      = "RELIANCE.NS_Close"
WINDOW_LEN      = 32
HOP             = 8
PREDICT_HORIZON = 1
EPOCHS          = 50
BATCH_SIZE      = 32
# ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STOCK FORECAST — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Download and normalize
    print("\n[1/4] Downloading stock data...")
    df_raw = download_stock_data(TICKERS, START_DATE, END_DATE)
    print(f"       Shape: {df_raw.shape}  |  Columns: {len(df_raw.columns)}")

    df_norm, scalers = normalize(df_raw)
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    print("       Scalers saved to scalers.pkl")

    # 2. Build dataset
    print("\n[2/4] Building STFT spectrogram dataset...")
    X, y = make_dataset(df_norm, TARGET_COL, WINDOW_LEN, HOP, PREDICT_HORIZON)
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(X, y)
    print(f"       Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")

    # 3. Build and train model
    print("\n[3/4] Building CNN...")
    model = build_cnn(input_shape=X_tr.shape[1:])
    model = compile_model(model)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    print("\n[3/4] Training...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # 4. Save everything
    print("\n[4/4] Saving artifacts...")
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/stock_cnn.keras")
    np.save("history.npy", history.history)
    np.save("X_test.npy", X_te)
    np.save("y_test.npy", y_te)

    # Print final metrics
    val_loss = min(history.history["val_loss"])
    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best validation MSE: {val_loss:.6f}")
    print(f"  Artifacts saved: saved_model/stock_cnn.keras, history.npy, scalers.pkl")
    print(f"  Next: python evaluate.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
