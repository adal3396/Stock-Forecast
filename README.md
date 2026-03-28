# 📈 Stock Price Forecasting — STFT + CNN

A **Pattern Recognition for Financial Time Series Forecasting** web application that combines Short-Time Fourier Transform (STFT) signal processing with Convolutional Neural Networks (CNN) to predict stock prices.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)

## 🚀 Features

- **Data Pipeline** — Downloads OHLCV data from Yahoo Finance for Indian stocks (RELIANCE, TCS, INFY) + macro signals (Sensex, USD-INR)
- **Signal Processing** — FFT frequency spectrum + STFT spectrogram analysis with configurable window/hop parameters
- **CNN Model** — 3-block convolutional architecture for regression-based price prediction
- **Interactive Dashboard** — 4-tab Streamlit webapp with Plotly charts, metrics, and ablation study
- **Ablation Study** — Measures feature importance by selectively removing data sources

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Data | Download & visualize raw stock prices |
| 🌊 Spectrogram | FFT spectrum + STFT heatmap visualization |
| 🤖 Prediction | CNN inference with predicted vs actual + error distribution |
| 📐 Evaluation | Training curves, ablation study, generated figures gallery |

## 🛠️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/adal3396/Stock-Forecast.git
cd Stock-Forecast

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

### 1. Train the Model
```bash
python train.py
```
Downloads stock data, builds STFT spectrograms, trains the CNN (~2 min).

### 2. Generate Figures & Ablation Study
```bash
python evaluate.py
```
Generates all required figures in `figures/` directory.

### 3. Launch the Dashboard
```bash
python -m streamlit run app.py
```
Opens at `http://localhost:8501`.

## 📁 Project Structure

```
stock_forecast/
├── preprocess.py       ← Data download, normalization, STFT, dataset building
├── model.py            ← CNN architecture (3 conv blocks + regression head)
├── train.py            ← Training pipeline with EarlyStopping + LR scheduling
├── evaluate.py         ← All figures + ablation study
├── app.py              ← Streamlit webapp (4 tabs)
├── requirements.txt    ← Python dependencies
├── saved_model/        ← Created after training
├── figures/            ← Created after evaluation
└── data/               ← Optional local CSV cache
```

## 🧠 Architecture

```
Input (F × T_frames × C)
    → Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    → Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    → Conv2D(128, 3×3) + BatchNorm + ReLU
    → GlobalAveragePooling2D
    → Dense(128) + Dropout(0.3)
    → Dense(64) + ReLU
    → Dense(1)  ← Predicted price
```

## 📚 Methodology

1. **Data Preparation** — OHLCV + macro signals, MinMax normalization
2. **STFT** — Sliding window (L=32, H=8) with Hann window → 2D spectrograms
3. **CNN Training** — 50 epochs, EarlyStopping, ReduceLROnPlateau
4. **Evaluation** — MSE, MAE, R² metrics + ablation study

## 📜 License

MIT License
