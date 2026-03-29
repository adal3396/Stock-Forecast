# Stock Price Forecasting — STFT + CNN

**Pattern Recognition for Financial Time Series Forecasting** — STFT spectrograms + a CNN regressor, with a **Next.js** dashboard backed by **[Supabase](https://supabase.com)** (Postgres), deployable on [Vercel](https://vercel.com).

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)

## Features

- **Data pipeline** — Yahoo Finance OHLCV (e.g. RELIANCE, TCS, INFY) + Sensex and USD-INR
- **Signal processing** — FFT + STFT spectrograms
- **CNN** — Three conv blocks + regression head
- **Web dashboard** — Four tabs in `web/`, data loaded from Supabase `dashboard_documents`
- **Ablation study** — `evaluate.py` → CSV → `upload_to_supabase.py`

## Supabase setup (once)

1. Open your project → **SQL Editor**, paste and run `supabase_schema.sql` from this folder.
2. Copy **Project URL** and the **service_role** key (**Project Settings → API**).  
   The Postgres **database password** is only for direct DB tools; this app uses the **service_role** JWT for the API and upload script.

## Python setup

```bash
git clone https://github.com/adal3396/Stock-Forecast.git
cd Stock-Forecast

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

1. **Train** — `python train.py`
2. **Figures + ablation** — `python evaluate.py`
3. **Upload to Supabase** (from trained artifacts):

   ```powershell
   $env:SUPABASE_SERVICE_ROLE_KEY = "eyJhbGci...service_role..."
   $env:NEXT_PUBLIC_SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
   python upload_to_supabase.py
   ```

4. **Dashboard locally**

   ```bash
   cd web
   copy .env.example .env.local
   rem Fill in SUPABASE_SERVICE_ROLE_KEY and URL
   npm install
   npm run dev
   ```

## Deploy on Vercel

1. **Root directory:** `web`
2. **Environment variables**
   - `NEXT_PUBLIC_SUPABASE_URL` = your Supabase URL  
   - `SUPABASE_SERVICE_ROLE_KEY` = service role secret (server-only)

## Project layout

```
├── supabase_schema.sql # Run in Supabase SQL Editor
├── preprocess.py
├── model.py
├── train.py
├── evaluate.py
├── upload_to_supabase.py
├── requirements.txt
├── web/
│   ├── lib/supabase-server.js
│   └── app/api/dashboard/route.js
└── ...
```

## License

MIT License
