"""
upload_to_mongo.py — Upload all computed results to MongoDB for the Vercel web app.

Reads the locally-trained model outputs and uploads them as JSON documents
to MongoDB so the Next.js frontend can display them.

Set the connection string in the environment (never commit secrets):

  Windows PowerShell:
    $env:MONGODB_URI = "mongodb+srv://USER:PASS@cluster.example.net/"
  macOS/Linux:
    export MONGODB_URI='mongodb+srv://USER:PASS@cluster.example.net/'
"""

import os
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from preprocess import download_stock_data, normalize, compute_stft, make_dataset, train_val_test_split

MONGO_URI = os.environ.get("MONGODB_URI")
if not MONGO_URI:
    raise SystemExit(
        "MONGODB_URI is not set. Set it to your MongoDB Atlas connection string and retry."
    )

DB_NAME   = "stock_forecast"

TICKERS    = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
TARGET_COL = "RELIANCE.NS_Close"
TICKER     = "RELIANCE.NS"
WINDOW_LEN = 32
HOP        = 8


def main():
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db["dashboard"]

    # Clear old data
    coll.delete_many({})
    print("Cleared old data.")

    # ── Download & normalize data ────────────────────────────────────
    print("Downloading stock data...")
    df_raw = download_stock_data(TICKERS, START_DATE, END_DATE)
    df_norm, _ = normalize(df_raw)

    # Load scalers
    with open("scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # ── 1. Stock Prices ──────────────────────────────────────────────
    print("Uploading stock prices...")
    close_cols = [c for c in df_raw.columns if "_Close" in c]
    stock_data = {
        "type": "stock_prices",
        "dates": [d.strftime("%Y-%m-%d") for d in df_raw.index],
        "tickers": {}
    }
    for col in close_cols:
        ticker_name = col.replace("_Close", "")
        stock_data["tickers"][ticker_name] = [
            round(float(v), 2) for v in df_raw[col].values
        ]
    coll.insert_one(stock_data)

    # ── 2. FFT Data ──────────────────────────────────────────────────
    print("Uploading FFT data...")
    raw_signal = df_raw[f"{TICKER}_Close"].values
    fft_vals   = np.fft.rfft(raw_signal)
    fft_freqs  = np.fft.rfftfreq(len(raw_signal), d=1.0)
    magnitude  = np.abs(fft_vals)

    fft_data = {
        "type": "fft",
        "ticker": TICKER,
        "frequencies": [round(float(f), 6) for f in fft_freqs],
        "magnitudes":  [round(float(m), 2) for m in magnitude],
    }
    coll.insert_one(fft_data)

    # ── 3. Spectrogram Data ──────────────────────────────────────────
    print("Uploading spectrogram data...")
    signal_norm        = df_norm[TARGET_COL].values
    freqs, times, Sxx  = compute_stft(signal_norm, WINDOW_LEN, HOP)
    power_db           = 10 * np.log10(Sxx + 1e-10)

    spec_data = {
        "type": "spectrogram",
        "ticker": TICKER,
        "frequencies": [round(float(f), 6) for f in freqs],
        "times":       [round(float(t), 4) for t in times],
        "power_db":    [[round(float(v), 2) for v in row] for row in power_db],
        "window_len":  WINDOW_LEN,
        "hop":         HOP,
    }
    coll.insert_one(spec_data)

    # ── 4. Predictions ───────────────────────────────────────────────
    print("Uploading predictions...")
    import tensorflow as tf
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    X, y = make_dataset(df_norm, TARGET_COL, WINDOW_LEN, HOP)
    _, _, X_te, _, _, y_te = train_val_test_split(X, y)

    model  = tf.keras.models.load_model("saved_model/stock_cnn.keras")
    y_pred = model.predict(X_te).flatten()

    sc        = scalers[f"{TICKER}_Close"]
    actual    = sc.inverse_transform(y_te.reshape(-1, 1)).flatten()
    predicted = sc.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = float(mean_squared_error(actual, predicted))
    mae = float(mean_absolute_error(actual, predicted))
    r2  = float(r2_score(actual, predicted))

    errors = (actual - predicted).tolist()

    pred_data = {
        "type": "predictions",
        "ticker": TICKER,
        "actual":    [round(float(v), 2) for v in actual],
        "predicted": [round(float(v), 2) for v in predicted],
        "errors":    [round(float(v), 2) for v in errors],
    }
    coll.insert_one(pred_data)

    # ── 5. Metrics ───────────────────────────────────────────────────
    print("Uploading metrics...")
    metrics_data = {
        "type": "metrics",
        "mse":  round(mse, 2),
        "mae":  round(mae, 2),
        "r2":   round(r2, 4),
        "trading_days": len(df_raw),
        "features": len(df_raw.columns),
        "companies": len(TICKERS),
        "tickers": TICKERS,
    }
    coll.insert_one(metrics_data)

    # ── 6. Training History ──────────────────────────────────────────
    print("Uploading training history...")
    history = np.load("history.npy", allow_pickle=True).item()
    history_data = {
        "type": "training_history",
        "loss":     [round(float(v), 6) for v in history["loss"]],
        "val_loss": [round(float(v), 6) for v in history["val_loss"]],
    }
    if "mae" in history:
        history_data["mae"]     = [round(float(v), 6) for v in history["mae"]]
        history_data["val_mae"] = [round(float(v), 6) for v in history["val_mae"]]
    coll.insert_one(history_data)

    # ── 7. Ablation Results ──────────────────────────────────────────
    print("Uploading ablation results...")
    df_abl = pd.read_csv("ablation_results.csv")
    ablation_data = {
        "type": "ablation",
        "experiments": df_abl.to_dict(orient="records"),
    }
    coll.insert_one(ablation_data)

    print(f"\n{'=' * 50}")
    print(f"  ✅ All data uploaded to MongoDB!")
    print(f"  Database: {DB_NAME}")
    print(f"  Collection: dashboard")
    print(f"  Documents: 7")
    print(f"{'=' * 50}")

    client.close()


if __name__ == "__main__":
    main()
