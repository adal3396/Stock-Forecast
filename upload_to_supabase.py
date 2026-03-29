"""
upload_to_supabase.py — Upload computed results to Supabase (Postgres) for the Next.js app.

Requires:
  - Table public.dashboard_documents (see supabase_schema.sql in this folder).
  - Environment variables (never commit secrets):

      SUPABASE_SERVICE_ROLE_KEY   ← Supabase → Project Settings → API → service_role
      SUPABASE_URL                 optional; defaults to NEXT_PUBLIC_SUPABASE_URL

  PowerShell example:

      $env:SUPABASE_SERVICE_ROLE_KEY = "eyJhbGci..."
      $env:NEXT_PUBLIC_SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
      python upload_to_supabase.py

  The Postgres *database password* is not used by this script; use the service_role JWT.
"""

import os
import pickle
import numpy as np
import pandas as pd
from supabase import create_client
from preprocess import download_stock_data, normalize, compute_stft, make_dataset, train_val_test_split

TABLE = "dashboard_documents"

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

if not SUPABASE_URL:
    raise SystemExit(
        "Set SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL to your Supabase project URL."
    )
if not SUPABASE_KEY:
    raise SystemExit(
        "Set SUPABASE_SERVICE_ROLE_KEY (Project Settings → API → service_role secret)."
    )

TICKERS    = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
TARGET_COL = "RELIANCE.NS_Close"
TICKER     = "RELIANCE.NS"
WINDOW_LEN = 32
HOP        = 8


def upsert(supabase, doc_type, payload):
    supabase.table(TABLE).upsert(
        {"doc_type": doc_type, "payload": payload},
        on_conflict="doc_type",
    ).execute()


def main():
    print("Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Downloading stock data...")
    df_raw = download_stock_data(TICKERS, START_DATE, END_DATE)
    df_norm, _ = normalize(df_raw)

    with open("scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    print("Uploading stock prices...")
    close_cols = [c for c in df_raw.columns if "_Close" in c]
    tickers = {}
    for col in close_cols:
        ticker_name = col.replace("_Close", "")
        tickers[ticker_name] = [round(float(v), 2) for v in df_raw[col].values]
    upsert(
        supabase,
        "stock_prices",
        {
            "dates": [d.strftime("%Y-%m-%d") for d in df_raw.index],
            "tickers": tickers,
        },
    )

    print("Uploading FFT...")
    raw_signal = df_raw[f"{TICKER}_Close"].values
    fft_vals = np.fft.rfft(raw_signal)
    fft_freqs = np.fft.rfftfreq(len(raw_signal), d=1.0)
    magnitude = np.abs(fft_vals)
    upsert(
        supabase,
        "fft",
        {
            "ticker": TICKER,
            "frequencies": [round(float(f), 6) for f in fft_freqs],
            "magnitudes": [round(float(m), 2) for m in magnitude],
        },
    )

    print("Uploading spectrogram...")
    signal_norm = df_norm[TARGET_COL].values
    freqs, times, Sxx = compute_stft(signal_norm, WINDOW_LEN, HOP)
    power_db = 10 * np.log10(Sxx + 1e-10)
    upsert(
        supabase,
        "spectrogram",
        {
            "ticker": TICKER,
            "frequencies": [round(float(f), 6) for f in freqs],
            "times": [round(float(t), 4) for t in times],
            "power_db": [[round(float(v), 2) for v in row] for row in power_db],
            "window_len": WINDOW_LEN,
            "hop": HOP,
        },
    )

    print("Uploading predictions...")
    import tensorflow as tf
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    X, y = make_dataset(df_norm, TARGET_COL, WINDOW_LEN, HOP)
    _, _, X_te, _, _, y_te = train_val_test_split(X, y)

    model = tf.keras.models.load_model("saved_model/stock_cnn.keras")
    y_pred = model.predict(X_te).flatten()

    sc = scalers[f"{TICKER}_Close"]
    actual = sc.inverse_transform(y_te.reshape(-1, 1)).flatten()
    predicted = sc.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = float(mean_squared_error(actual, predicted))
    mae = float(mean_absolute_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    errors = (actual - predicted).tolist()

    upsert(
        supabase,
        "predictions",
        {
            "ticker": TICKER,
            "actual": [round(float(v), 2) for v in actual],
            "predicted": [round(float(v), 2) for v in predicted],
            "errors": [round(float(v), 2) for v in errors],
        },
    )

    print("Uploading metrics...")
    upsert(
        supabase,
        "metrics",
        {
            "mse": round(mse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "trading_days": len(df_raw),
            "features": len(df_raw.columns),
            "companies": len(TICKERS),
            "tickers": TICKERS,
        },
    )

    print("Uploading training history...")
    history = np.load("history.npy", allow_pickle=True).item()
    hist_payload = {
        "loss": [round(float(v), 6) for v in history["loss"]],
        "val_loss": [round(float(v), 6) for v in history["val_loss"]],
    }
    if "mae" in history:
        hist_payload["mae"] = [round(float(v), 6) for v in history["mae"]]
        hist_payload["val_mae"] = [round(float(v), 6) for v in history["val_mae"]]
    upsert(supabase, "training_history", hist_payload)

    print("Uploading ablation...")
    df_abl = pd.read_csv("ablation_results.csv")
    upsert(
        supabase,
        "ablation",
        {"experiments": df_abl.to_dict(orient="records")},
    )

    print(f"\n{'=' * 50}")
    print("  ✅ All data uploaded to Supabase (table dashboard_documents).")
    print(f"  Rows upserted: 7 doc_types")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
