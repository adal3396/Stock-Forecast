"""
evaluate.py — Phase 4: Evaluation & Analysis

Generates all required figures:
  Figure 1: Stock closing prices over time
  Figure 2: Frequency spectrum (FFT)
  Figure 3: STFT spectrogram
  Figure 4: CNN architecture diagram
  Figure 5: Predicted vs Actual prices with metrics

Also runs an ablation study to measure feature importance.

Run:  python evaluate.py   (after train.py)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import (
    download_stock_data, normalize,
    make_dataset, train_val_test_split, compute_stft
)
from model import build_cnn, compile_model

# ── CONFIG (must match train.py) ────────────────────────────────────
TICKERS    = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
TARGET_COL = "RELIANCE.NS_Close"
TICKER     = "RELIANCE.NS"
WINDOW_LEN = 32
HOP        = 8
# ────────────────────────────────────────────────────────────────────


def main():
    os.makedirs("figures", exist_ok=True)

    print("Loading data and model...")
    df_raw           = download_stock_data(TICKERS, START_DATE, END_DATE)
    df_norm, _       = normalize(df_raw)
    with open("scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    X, y             = make_dataset(df_norm, TARGET_COL, WINDOW_LEN, HOP)
    _, _, X_te, _, _, y_te = train_val_test_split(X, y)
    model            = tf.keras.models.load_model("saved_model/stock_cnn.keras")
    y_pred           = model.predict(X_te).flatten()

    # ── Figure 1: Time Series ────────────────────────────────────────
    print("Generating Figure 1: Time series plot...")
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in TICKERS:
        col = f"{ticker}_Close"
        ax.plot(df_raw.index, df_raw[col], label=ticker, linewidth=1.2)
    ax.set_title("Stock Closing Prices Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig1_time_series.png", dpi=150)
    plt.close()

    # ── Figure 2: Frequency Spectrum (FFT) ───────────────────────────
    print("Generating Figure 2: FFT spectrum...")
    signal    = df_raw[f"{TICKER}_Close"].values
    fft_vals  = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal), d=1.0)
    magnitude = np.abs(fft_vals)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(fft_freqs, magnitude, color="steelblue", linewidth=0.8)
    ax.set_title(f"Frequency Spectrum — {TICKER}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig2_fft_spectrum.png", dpi=150)
    plt.close()

    # ── Figure 3: Spectrogram (STFT) ─────────────────────────────────
    print("Generating Figure 3: STFT spectrogram...")
    signal_norm       = df_norm[TARGET_COL].values
    freqs, times, Sxx = compute_stft(signal_norm, WINDOW_LEN, HOP)

    fig, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(
        times, freqs,
        10 * np.log10(Sxx + 1e-10),
        cmap="inferno", shading="auto"
    )
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    ax.set_title(f"STFT Spectrogram — {TICKER}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency (cycles/day)")
    plt.tight_layout()
    plt.savefig("figures/fig3_spectrogram.png", dpi=150)
    plt.close()

    # ── Figure 4: CNN Architecture ───────────────────────────────────
    print("Generating Figure 4: CNN architecture...")
    try:
        arch_model = build_cnn(X_te.shape[1:])
        compile_model(arch_model)
        tf.keras.utils.plot_model(
            arch_model, to_file="figures/fig4_cnn_arch.png",
            show_shapes=True, show_layer_names=True, dpi=96
        )
    except Exception as e:
        print(f"  ⚠ Could not generate architecture diagram: {e}")
        print("  (Install pydot and graphviz for this figure)")

    # ── Figure 5: Predicted vs Actual ────────────────────────────────
    print("Generating Figure 5: Predictions...")
    sc        = scalers[f"{TICKER}_Close"]
    actual    = sc.inverse_transform(y_te.reshape(-1, 1)).flatten()
    predicted = sc.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2  = r2_score(actual, predicted)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual,    label="Actual",    linewidth=1.2, color="steelblue")
    ax.plot(predicted, label="Predicted", linewidth=1.2, color="darkorange")
    ax.set_title(f"Predicted vs Actual — {TICKER}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.01, 0.95,
            f"MSE={mse:,.2f}  MAE={mae:,.2f}  R²={r2:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig("figures/fig5_predictions.png", dpi=150)
    plt.close()

    print(f"\n{'─' * 50}")
    print(f"  Test MSE : {mse:,.4f}")
    print(f"  Test MAE : {mae:,.4f}")
    print(f"  Test R²  : {r2:.4f}")
    print(f"{'─' * 50}")

    # ── Ablation Study ───────────────────────────────────────────────
    print("\nRunning ablation study...")
    ablation_study(df_norm, scalers, TICKERS, TARGET_COL, WINDOW_LEN, HOP)

    print("\n✅ All figures saved to figures/ directory.")
    print("   Next: streamlit run app.py")


def ablation_study(df_norm, scalers, tickers, target_col, window_len, hop):
    """
    Train a new model for each experiment.
    Measures how much each feature group contributes to prediction accuracy.
    """
    experiments = {
        "All features": list(df_norm.columns),
    }
    for ticker in tickers:
        remaining = [c for c in df_norm.columns if ticker not in c]
        experiments[f"Remove {ticker}"] = remaining

    macro = [c for c in df_norm.columns if c in ("Sensex", "USD_INR")]
    if macro:
        experiments["Remove macro"] = [c for c in df_norm.columns if c not in macro]

    results = []
    for name, cols in experiments.items():
        print(f"  Running: {name} ({len(cols)} features)...")
        sub = df_norm[cols]
        tc  = target_col if target_col in cols else cols[0]
        X, y = make_dataset(sub, tc, window_len, hop)
        X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(X, y)

        m = build_cnn(X_tr.shape[1:])
        compile_model(m)
        m.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=20, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

        y_p = m.predict(X_te, verbose=0).flatten()
        mse = mean_squared_error(y_te, y_p)
        results.append({"Experiment": name, "Features": len(cols), "Test MSE": mse})
        print(f"    → MSE = {mse:.6f}")

    df_results = pd.DataFrame(results).sort_values("Test MSE")
    df_results.to_csv("ablation_results.csv", index=False)
    print("\n  Ablation results:")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
