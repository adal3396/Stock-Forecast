"""
app.py — Streamlit Webapp for Stock Price Forecasting

A 4-tab interactive dashboard:
  Tab 1: 📊 Data         — download and visualize raw prices
  Tab 2: 🌊 Spectrogram  — FFT spectrum + STFT heatmap
  Tab 3: 🤖 Prediction   — CNN inference, predicted vs actual
  Tab 4: 📐 Evaluation   — training curves, ablation table

Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# ── Must be the first Streamlit command ──────────────────────────────
st.set_page_config(
    page_title="Stock Forecast — STFT + CNN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Import project modules ──────────────────────────────────────────
from preprocess import download_stock_data, normalize, compute_stft, make_dataset, train_val_test_split

# ── Custom CSS for a polished look ──────────────────────────────────
st.markdown("""
<style>
    /* Global dark-ish theme tweaks */
    .main .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #a0aec0 !important;
        font-size: 0.85rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* Hero header */
    .hero-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #cbd5e0;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📈 Stock Price Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Time–Frequency Analysis with STFT + CNN Deep Learning</div>', unsafe_allow_html=True)


# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    tickers_input = st.text_input(
        "🏢 Tickers (comma-separated)",
        "RELIANCE.NS, TCS.NS, INFY.NS",
        help="Yahoo Finance ticker symbols for Indian stocks"
    )
    TICKERS = [t.strip() for t in tickers_input.split(",")]

    st.markdown("##### 📅 Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        start = st.date_input("Start", value=pd.to_datetime("2018-01-01"))
    with col_e:
        end = st.date_input("End", value=pd.to_datetime("2024-01-01"))

    target = st.selectbox("🎯 Company to predict", TICKERS)

    st.markdown("##### 🔧 STFT Parameters")
    L = st.slider("Window length L", 16, 128, 32, 8,
                  help="Controls frequency vs time resolution trade-off")
    H = st.slider("Hop size H", 2, 32, 8, 2,
                  help="Step between windows (overlap = L − H)")

    horizon = st.selectbox("📆 Predict N days ahead", [1, 3, 5])

    st.markdown("---")
    run_btn = st.button("🚀 Load Data", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#718096; font-size:0.8rem; text-align:center;">
        Pattern Recognition for<br>Financial Time Series Forecasting<br>
        <strong>Assignment 2</strong>
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data",
    "🌊 Spectrogram",
    "🤖 Prediction",
    "📐 Evaluation"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Raw Stock Prices")
    st.markdown('<div class="info-box">📌 Click <strong>Load Data</strong> in the sidebar to download stock data from Yahoo Finance.</div>', unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Downloading stock data from Yahoo Finance..."):
            df_raw = download_stock_data(TICKERS, str(start), str(end))
            st.session_state["df_raw"] = df_raw
            st.success(f"✅ Downloaded {len(df_raw)} trading days × {len(df_raw.columns)} features")

    if "df_raw" in st.session_state:
        df = st.session_state["df_raw"]
        close_cols = [c for c in df.columns if "_Close" in c]

        # Interactive plotly chart
        fig = go.Figure()
        colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181"]
        for i, col in enumerate(close_cols):
            ticker_name = col.replace("_Close", "")
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=ticker_name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{ticker_name}</b><br>Date: %{{x}}<br>Price: ₹%{{y:,.2f}}<extra></extra>"
            ))
        fig.update_layout(
            title=dict(text="Stock Closing Prices Over Time", font=dict(size=18)),
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_dark",
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data summary
        c1, c2, c3 = st.columns(3)
        c1.metric("📅 Trading Days", f"{len(df):,}")
        c2.metric("📊 Features", f"{len(df.columns)}")
        c3.metric("🏢 Companies", f"{len(TICKERS)}")

        # Raw data table
        with st.expander("📋 View raw data (last 50 rows)"):
            st.dataframe(df.tail(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — SPECTROGRAM
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Frequency Analysis")

    if "df_raw" in st.session_state:
        df_raw = st.session_state["df_raw"]
        df_norm, scalers = normalize(df_raw)
        st.session_state["df_norm"] = df_norm
        st.session_state["scalers"] = scalers

        target_col = f"{target}_Close"
        signal = df_norm[target_col].values

        col_a, col_b = st.columns(2)

        # Figure 2: FFT
        with col_a:
            st.markdown("#### Figure 2 — Frequency Spectrum (FFT)")
            raw_signal = df_raw[target_col].values
            fft_vals  = np.fft.rfft(raw_signal)
            fft_freqs = np.fft.rfftfreq(len(raw_signal), d=1.0)
            magnitude = np.abs(fft_vals)

            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=fft_freqs, y=magnitude,
                mode="lines",
                line=dict(color="#667eea", width=1),
                fill="tozeroy",
                fillcolor="rgba(102,126,234,0.2)"
            ))
            fig_fft.update_layout(
                xaxis_title="Frequency (cycles/day)",
                yaxis_title="Magnitude",
                template="plotly_dark",
                height=350,
                margin=dict(t=20)
            )
            st.plotly_chart(fig_fft, use_container_width=True)

            st.markdown('<div class="info-box">🔍 The FFT decomposes the signal into frequency components. Peaks indicate dominant cycles in the stock price.</div>', unsafe_allow_html=True)

        # Figure 3: Spectrogram
        with col_b:
            st.markdown("#### Figure 3 — STFT Spectrogram")
            freqs, times, Sxx = compute_stft(signal, L, H)
            power_db = 10 * np.log10(Sxx + 1e-10)

            fig_spec, ax = plt.subplots(figsize=(8, 4))
            pcm = ax.pcolormesh(times, freqs, power_db,
                                cmap="inferno", shading="auto")
            fig_spec.colorbar(pcm, ax=ax, label="Power (dB)")
            ax.set_xlabel("Time frame")
            ax.set_ylabel("Frequency (cycles/day)")
            ax.set_title(f"STFT Spectrogram — {target}", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_spec)
            plt.close()

            st.markdown('<div class="info-box">🌊 The spectrogram shows how frequency content evolves over time. Bright bands = strong periodic patterns. Low freq = long-term trends, high freq = short-term noise.</div>', unsafe_allow_html=True)

        # STFT parameters info
        st.markdown("---")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Window L", f"{L} days")
        p2.metric("Hop H", f"{H} days")
        p3.metric("Overlap", f"{L - H} days")
        p4.metric("Freq bins", f"{len(freqs)}")
    else:
        st.info("👆 Load data first using the sidebar button.")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTION
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("CNN Price Prediction")

    # Training parameters — must match what train.py used
    TRAIN_L = 32
    TRAIN_H = 8
    TRAIN_HORIZON = 1

    if "df_raw" not in st.session_state:
        st.info("👆 Load data first using the sidebar button.")
    elif not os.path.exists("saved_model/stock_cnn.keras"):
        st.warning("⚠️ No trained model found. Run `python train.py` first to train the CNN.")
    else:
        st.markdown(f'<div class="info-box">ℹ️ Prediction uses the <strong>trained model\'s parameters</strong> (L={TRAIN_L}, H={TRAIN_H}) to ensure input shape compatibility. The spectrogram tab uses the sidebar sliders for exploration.</div>', unsafe_allow_html=True)

        if st.button("🤖 Load Model & Predict", type="primary"):
            with st.spinner("Running CNN inference..."):
                try:
                    import tensorflow as tf
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                    df_raw   = st.session_state["df_raw"]
                    df_norm, scalers = normalize(df_raw)
                    TARGET_COL = f"{target}_Close"

                    # Use TRAINING parameters, not sidebar sliders
                    X, y = make_dataset(df_norm, TARGET_COL, TRAIN_L, TRAIN_H, TRAIN_HORIZON)
                    _, _, X_te, _, _, y_te = train_val_test_split(X, y)

                    model  = tf.keras.models.load_model("saved_model/stock_cnn.keras")
                    y_pred = model.predict(X_te).flatten()

                    sc     = scalers[TARGET_COL]
                    actual = sc.inverse_transform(y_te.reshape(-1, 1)).flatten()
                    pred   = sc.inverse_transform(y_pred.reshape(-1, 1)).flatten()

                    # Metrics
                    mse_val = mean_squared_error(actual, pred)
                    mae_val = mean_absolute_error(actual, pred)
                    r2_val  = r2_score(actual, pred)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("📉 Test MSE", f"{mse_val:,.2f}")
                    c2.metric("📏 Test MAE", f"{mae_val:,.2f}")
                    c3.metric("📊 R² Score", f"{r2_val:.4f}")

                    # Interactive prediction chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=actual, name="Actual",
                        line=dict(color="#667eea", width=2),
                        hovertemplate="Actual: ₹%{y:,.2f}<extra></extra>"
                    ))
                    fig.add_trace(go.Scatter(
                        y=pred, name="Predicted",
                        line=dict(color="#f093fb", width=2, dash="dot"),
                        hovertemplate="Predicted: ₹%{y:,.2f}<extra></extra>"
                    ))
                    fig.update_layout(
                        title=dict(text=f"Predicted vs Actual — {target}", font=dict(size=18)),
                        xaxis_title="Test Sample",
                        yaxis_title="Price (INR)",
                        template="plotly_dark",
                        height=450,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Error distribution
                    errors = actual - pred
                    fig_err = go.Figure()
                    fig_err.add_trace(go.Histogram(
                        x=errors, nbinsx=30,
                        marker_color="#4fd1c5",
                        opacity=0.8
                    ))
                    fig_err.update_layout(
                        title="Prediction Error Distribution",
                        xaxis_title="Error (INR)",
                        yaxis_title="Count",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig_err, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.info("💡 Make sure the data tickers and date range match what was used during training (RELIANCE.NS, TCS.NS, INFY.NS, 2018-2024).")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Metrics & Ablation Study")

    col_left, col_right = st.columns(2)

    # Training loss curves
    with col_left:
        st.markdown("#### Training & Validation Loss")
        try:
            history = np.load("history.npy", allow_pickle=True).item()
            epochs = list(range(1, len(history["loss"]) + 1))

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=history["loss"],
                name="Train MSE",
                line=dict(color="#667eea", width=2),
                fill="tozeroy",
                fillcolor="rgba(102,126,234,0.15)"
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=history["val_loss"],
                name="Val MSE",
                line=dict(color="#f093fb", width=2),
                fill="tozeroy",
                fillcolor="rgba(240,147,251,0.15)"
            ))
            fig_loss.update_layout(
                xaxis_title="Epoch",
                yaxis_title="MSE Loss",
                template="plotly_dark",
                height=350,
                margin=dict(t=20)
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # MAE curves if available
            if "mae" in history:
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(
                    x=epochs, y=history["mae"],
                    name="Train MAE", line=dict(color="#4fd1c5", width=2)
                ))
                fig_mae.add_trace(go.Scatter(
                    x=epochs, y=history["val_mae"],
                    name="Val MAE", line=dict(color="#f6ad55", width=2)
                ))
                fig_mae.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title="MAE",
                    template="plotly_dark",
                    height=300,
                    margin=dict(t=20)
                )
                st.plotly_chart(fig_mae, use_container_width=True)
        except FileNotFoundError:
            st.info("📝 Run `python train.py` first to generate training history.")

    # Ablation results table
    with col_right:
        st.markdown("#### Ablation Study — Feature Importance")
        try:
            df_abl = pd.read_csv("ablation_results.csv")
            st.dataframe(df_abl, use_container_width=True, hide_index=True)

            # Bar chart
            fig_abl = go.Figure()
            fig_abl.add_trace(go.Bar(
                x=df_abl["Experiment"],
                y=df_abl["Test MSE"],
                marker_color=["#667eea" if i == 0 else "#f093fb"
                              for i in range(len(df_abl))],
                text=[f"{v:.6f}" for v in df_abl["Test MSE"]],
                textposition="outside"
            ))
            fig_abl.update_layout(
                yaxis_title="Test MSE",
                template="plotly_dark",
                height=350,
                margin=dict(t=20)
            )
            st.plotly_chart(fig_abl, use_container_width=True)

            st.markdown('<div class="info-box">📊 The ablation study compares model performance when different feature groups are removed. Higher MSE = that feature group was more important.</div>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.info("📝 Run `python evaluate.py` to generate ablation results.")

    # Saved figures gallery
    st.markdown("---")
    st.markdown("#### 📸 Generated Figures")
    figures_dir = "figures"
    if os.path.exists(figures_dir):
        fig_files = sorted([f for f in os.listdir(figures_dir) if f.endswith(".png")])
        if fig_files:
            cols = st.columns(min(len(fig_files), 3))
            for i, fname in enumerate(fig_files):
                with cols[i % 3]:
                    st.image(os.path.join(figures_dir, fname), caption=fname, use_container_width=True)
        else:
            st.info("No figures generated yet. Run `python evaluate.py`.")
    else:
        st.info("No figures directory found. Run `python evaluate.py`.")
