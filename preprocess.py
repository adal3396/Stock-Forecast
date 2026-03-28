"""
preprocess.py — Phase 1 & 2: Data download, normalization, STFT, dataset building.

This module handles:
  1. Downloading OHLCV stock data from Yahoo Finance (yfinance)
  2. Adding macro signals (Sensex, USD-INR exchange rate)
  3. Normalizing all features with MinMaxScaler
  4. Computing Short-Time Fourier Transform (STFT) spectrograms
  5. Building sliding-window CNN datasets from the spectrograms
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import stft as scipy_stft


# ─── Default Configuration ──────────────────────────────────────────────
TICKERS    = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
FEATURES   = ["Close", "Open", "High", "Low", "Volume"]


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — Data Preparation
# ═══════════════════════════════════════════════════════════════════════

def download_stock_data(tickers, start, end, include_macro=True):
    """
    Download OHLCV data for the given tickers from Yahoo Finance.

    Parameters
    ----------
    tickers       : list[str]  — Yahoo Finance ticker symbols
    start, end    : str        — date range in 'YYYY-MM-DD' format
    include_macro : bool       — whether to append Sensex & USD-INR

    Returns
    -------
    pd.DataFrame  — combined DataFrame, forward/back-filled, NaN-free
    """
    frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        # Handle multi-level columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[FEATURES].copy()
        df.columns = [f"{ticker}_{c}" for c in FEATURES]
        frames.append(df)

    combined = pd.concat(frames, axis=1)

    # Add macro signals
    if include_macro:
        sensex = yf.download("^BSESN", start=start, end=end,
                              auto_adjust=True, progress=False)
        usd_inr = yf.download("INR=X", start=start, end=end,
                               auto_adjust=True, progress=False)
        if isinstance(sensex.columns, pd.MultiIndex):
            sensex.columns = sensex.columns.get_level_values(0)
        if isinstance(usd_inr.columns, pd.MultiIndex):
            usd_inr.columns = usd_inr.columns.get_level_values(0)
        combined["Sensex"]  = sensex["Close"]
        combined["USD_INR"] = usd_inr["Close"]

    # Fill missing trading days
    combined.ffill(inplace=True)
    combined.bfill(inplace=True)
    combined.dropna(inplace=True)
    return combined


def normalize(df):
    """
    Min-Max normalize each column independently.

    Returns
    -------
    normed  : pd.DataFrame — normalized copy (values in [0, 1])
    scalers : dict[str, MinMaxScaler] — one scaler per column (for inverse_transform)
    """
    scalers = {}
    normed  = df.copy()
    for col in df.columns:
        sc = MinMaxScaler()
        normed[col] = sc.fit_transform(df[[col]])
        scalers[col] = sc
    return normed, scalers


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — Signal Processing (STFT)
# ═══════════════════════════════════════════════════════════════════════

def compute_stft(signal, window_len=32, hop=8):
    """
    Compute Short-Time Fourier Transform of a 1-D signal.

    Parameters
    ----------
    signal     : np.ndarray  — 1-D array (one column of time series)
    window_len : int         — L, samples per segment
    hop        : int         — H, step between windows

    Returns
    -------
    freqs : shape (F,)   — frequency axis, F = window_len // 2 + 1
    times : shape (T,)   — time-frame axis
    Sxx   : shape (F, T) — power spectrogram  |STFT|²
    """
    noverlap = window_len - hop
    freqs, times, Zxx = scipy_stft(
        signal, fs=1.0,
        nperseg=window_len,
        noverlap=noverlap,
        window="hann"
    )
    Sxx = np.abs(Zxx) ** 2   # magnitude squared = power spectrogram
    return freqs, times, Sxx


def make_dataset(normed_df, target_col, window_len=32, hop=8, predict_horizon=1):
    """
    Build CNN dataset by sliding a window over time.

    For each window position:
      1. Extract a patch of length `window_len`
      2. Compute STFT spectrogram for every feature column
      3. Stack spectrograms as channels → one sample

    Label = normalized Close price `predict_horizon` steps ahead.

    Returns
    -------
    X : np.ndarray, shape (N, F, T_frames, C)  — channels-last for Keras Conv2D
    y : np.ndarray, shape (N,)                  — future normalized price
    """
    n = len(normed_df)
    X_list, y_list = [], []

    for start in range(0, n - window_len - predict_horizon, hop):
        end   = start + window_len
        patch = normed_df.iloc[start:end]

        # Compute STFT for every feature column
        specs = []
        for col in normed_df.columns:
            _, _, Sxx = compute_stft(patch[col].values, window_len, hop)
            specs.append(Sxx)

        # Stack: (F, T_frames, C)
        spec_stack = np.stack(specs, axis=-1)
        X_list.append(spec_stack)

        future_idx = min(end + predict_horizon - 1, n - 1)
        y_list.append(normed_df[target_col].iloc[future_idx])

    X = np.array(X_list, dtype=np.float32)   # (N, F, T_frames, C)
    y = np.array(y_list, dtype=np.float32)   # (N,)
    return X, y


def train_val_test_split(X, y, train_frac=0.70, val_frac=0.15):
    """
    Chronological split — NEVER shuffle time series data.
    Split: 70 % train / 15 % validation / 15 % test
    """
    n  = len(X)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return X[:t1], X[t1:t2], X[t2:], y[:t1], y[t1:t2], y[t2:]


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Downloading sample data...")
    df = download_stock_data(TICKERS, START_DATE, END_DATE)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    df_n, sc = normalize(df)
    print(f"  Normalized range: [{df_n.min().min():.4f}, {df_n.max().max():.4f}]")

    X, y = make_dataset(df_n, "RELIANCE.NS_Close")
    print(f"  Dataset X: {X.shape}   y: {y.shape}")
