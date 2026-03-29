# Stock Price Forecasting — STFT + CNN

**Pattern Recognition for Financial Time Series Forecasting** — STFT spectrograms + a CNN regressor, with a **Next.js** dashboard backed by **MongoDB** (deployable on [Vercel](https://vercel.com)).

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)

## Features

- **Data pipeline** — Yahoo Finance OHLCV (e.g. RELIANCE, TCS, INFY) + Sensex and USD-INR
- **Signal processing** — FFT + STFT spectrograms (configurable window / hop in training)
- **CNN** — Three conv blocks + regression head for normalized close prediction
- **Web dashboard** — Four tabs (data, spectrogram, prediction, evaluation) served from `web/`, data from MongoDB
- **Ablation study** — Feature-removal experiments (`evaluate.py` → CSV → uploaded with `upload_to_mongo.py`)

## Python setup

```bash
git clone https://github.com/adal3396/Stock-Forecast.git
cd Stock-Forecast

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

1. **Train**

   ```bash
   python train.py
   ```

2. **Figures + ablation**

   ```bash
   python evaluate.py
   ```

3. **Upload results for the website** (requires Atlas connection string in env)

   ```bash
   set MONGODB_URI=mongodb+srv://USER:PASS@cluster.mongodb.net/
   python upload_to_mongo.py
   ```

4. **Run dashboard locally**

   ```bash
   cd web
   copy .env.example .env.local
   rem Edit .env.local: set MONGODB_URI
   npm install
   npm run dev
   ```

   Open [http://localhost:3000](http://localhost:3000).

## Deploy on Vercel

1. Import this repo in Vercel.
2. Set **Root Directory** to `web`.
3. Add environment variable **`MONGODB_URI`** (same Atlas URI as upload script).
4. Deploy.

Ensure Atlas **Network Access** allows connections from your hosting region (often `0.0.0.0/0` for coursework).

## Project layout

```
├── preprocess.py       # Data, normalization, STFT, dataset
├── model.py            # CNN
├── train.py            # Training
├── evaluate.py         # Figures + ablation CSV
├── upload_to_mongo.py # Pushes JSON docs to MongoDB for the UI
├── requirements.txt
├── web/                # Next.js app (Vercel)
│   ├── app/
│   ├── components/
│   └── lib/mongodb.js
├── saved_model/        # After train (gitignored)
└── figures/            # After evaluate (gitignored)
```

## License

MIT License
