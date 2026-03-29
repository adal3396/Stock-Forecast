"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, BarChart3, Database, LineChart as LineChartIcon, Loader2 } from "lucide-react";
import SpectrogramHeatmap from "./SpectrogramHeatmap";

const TABS = [
  { id: "data", label: "Data", icon: Database },
  { id: "spectrum", label: "Spectrogram", icon: Activity },
  { id: "prediction", label: "Prediction", icon: LineChartIcon },
  { id: "evaluation", label: "Evaluation", icon: BarChart3 },
];

const LINE_COLORS = ["#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b", "#ef4444"];

function Card({ children, className = "" }) {
  return (
    <div
      className={`rounded-xl border border-white/10 bg-white/[0.03] p-5 shadow-lg backdrop-blur ${className}`}
    >
      {children}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-lg border border-violet-500/20 bg-gradient-to-br from-violet-950/40 to-slate-900/60 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-100">{value}</div>
    </div>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState("data");
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancel = false;
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        const res = await fetch("/api/dashboard", { cache: "no-store" });
        const json = await res.json();
        if (cancel) return;
        if (!json.ok) {
          setErr(json.error || "Failed to load dashboard");
          setPayload(null);
        } else {
          setPayload(json.data || {});
        }
      } catch (e) {
        if (!cancel) setErr(e instanceof Error ? e.message : "Network error");
      } finally {
        if (!cancel) setLoading(false);
      }
    })();
    return () => {
      cancel = true;
    };
  }, []);

  const stockChartData = useMemo(() => {
    const sp = payload?.stock_prices;
    if (!sp?.dates?.length || !sp.tickers) return [];
    return sp.dates.map((d, i) => {
      const row = { date: d };
      for (const [name, series] of Object.entries(sp.tickers)) {
        row[name] = series[i] ?? null;
      }
      return row;
    });
  }, [payload]);

  const fftChartData = useMemo(() => {
    const fft = payload?.fft;
    if (!fft?.frequencies?.length) return [];
    return fft.frequencies.map((f, i) => ({
      f,
      mag: fft.magnitudes[i] ?? 0,
    }));
  }, [payload]);

  const predictionChartData = useMemo(() => {
    const p = payload?.predictions;
    if (!p?.actual?.length) return [];
    return p.actual.map((a, i) => ({
      i,
      actual: a,
      predicted: p.predicted[i] ?? null,
    }));
  }, [payload]);

  const historyChartData = useMemo(() => {
    const h = payload?.training_history;
    if (!h?.loss?.length) return [];
    return h.loss.map((train, i) => ({
      epoch: i + 1,
      train,
      val: h.val_loss[i] ?? null,
    }));
  }, [payload]);

  const ablationChartData = useMemo(() => {
    const a = payload?.ablation;
    if (!a?.experiments?.length) return [];
    return a.experiments.map((row) => ({
      name: row.Experiment || row.experiment || "",
      mse: row["Test MSE"] ?? row.test_mse ?? 0,
      features: row.Features ?? row.features,
    }));
  }, [payload]);

  const stockKeys = useMemo(() => {
    const sp = payload?.stock_prices?.tickers;
    return sp ? Object.keys(sp) : [];
  }, [payload]);

  if (loading) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3 text-slate-400">
        <Loader2 className="h-10 w-10 animate-spin text-violet-400" />
        <p>Loading dashboard from MongoDB…</p>
      </div>
    );
  }

  if (err) {
    return (
      <Card className="mx-auto max-w-xl border-red-500/30 bg-red-950/20">
        <h2 className="text-lg font-semibold text-red-200">Could not load data</h2>
        <p className="mt-2 text-sm text-red-200/80">{err}</p>
        <p className="mt-4 text-sm text-slate-400">
          Set <code className="rounded bg-black/40 px-1">MONGODB_URI</code> on Vercel, run{" "}
          <code className="rounded bg-black/40 px-1">python upload_to_mongo.py</code> from{" "}
          <code className="rounded bg-black/40 px-1">stock_forecast</code> after training.
        </p>
      </Card>
    );
  }

  const metrics = payload?.metrics;
  const spec = payload?.spectrogram;
  const preds = payload?.predictions;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2 border-b border-white/10 pb-4">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === id
                ? "bg-violet-600 text-white shadow-lg shadow-violet-600/25"
                : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      {tab === "data" && (
        <div className="space-y-4">
          {!stockChartData.length ? (
            <p className="text-slate-500">No stock price documents in MongoDB yet.</p>
          ) : (
            <>
              <div className="grid gap-3 sm:grid-cols-3">
                <Metric label="Trading days" value={metrics?.trading_days ?? "—"} />
                <Metric label="Features" value={metrics?.features ?? "—"} />
                <Metric label="Companies" value={metrics?.companies ?? stockKeys.length} />
              </div>
              <Card>
                <h3 className="mb-4 text-sm font-medium text-slate-300">Closing prices (INR)</h3>
                <div style={{ width: "100%", height: 420 }}>
                  <ResponsiveContainer>
                    <LineChart data={stockChartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                      <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} angle={-35} textAnchor="end" height={60} />
                      <YAxis tickFormatter={(v) => `₹${v}`} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ background: "#1e293b", border: "1px solid #475569" }}
                        labelStyle={{ color: "#e2e8f0" }}
                      />
                      <Legend />
                      {stockKeys.map((key, idx) => (
                        <Line
                          key={key}
                          type="monotone"
                          dataKey={key}
                          name={key.replace(".NS", "")}
                          stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                          dot={false}
                          strokeWidth={2}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </>
          )}
        </div>
      )}

      {tab === "spectrum" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <h3 className="mb-2 text-sm font-medium text-slate-300">
              FFT — {payload?.fft?.ticker ?? "target"}
            </h3>
            {!fftChartData.length ? (
              <p className="text-slate-500">No FFT document.</p>
            ) : (
              <div style={{ width: "100%", height: 320 }}>
                <ResponsiveContainer>
                  <LineChart data={fftChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                    <XAxis dataKey="f" tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: "cycles/day", fill: "#64748b", fontSize: 11, position: "bottom" }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569" }} />
                    <Line type="monotone" dataKey="mag" stroke="#8b5cf6" dot={false} strokeWidth={1.5} name="Magnitude" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </Card>
          <Card>
            <h3 className="mb-2 text-sm font-medium text-slate-300">
              STFT spectrogram — {spec?.ticker ?? "—"}
            </h3>
            {!spec?.power_db?.length ? (
              <p className="text-slate-500">No spectrogram document.</p>
            ) : (
              <div className="space-y-2">
                <SpectrogramHeatmap power_db={spec.power_db} width={640} height={300} />
                <p className="text-xs text-slate-500">
                  Window L={spec.window_len}, hop H={spec.hop} · Power (dB), inferno colormap
                </p>
              </div>
            )}
          </Card>
        </div>
      )}

      {tab === "prediction" && (
        <div className="space-y-4">
          {!predictionChartData.length ? (
            <p className="text-slate-500">No predictions in MongoDB. Run upload script after training.</p>
          ) : (
            <>
              <div className="grid gap-3 sm:grid-cols-3">
                <Metric label="Test MSE" value={metrics?.mse != null ? Number(metrics.mse).toLocaleString() : "—"} />
                <Metric label="Test MAE" value={metrics?.mae != null ? Number(metrics.mae).toLocaleString() : "—"} />
                <Metric label="R²" value={metrics?.r2 != null ? String(metrics.r2) : "—"} />
              </div>
              <Card>
                <h3 className="mb-4 text-sm font-medium text-slate-300">
                  Predicted vs actual — {preds?.ticker ?? ""}
                </h3>
                <div style={{ width: "100%", height: 400 }}>
                  <ResponsiveContainer>
                    <LineChart data={predictionChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                      <XAxis dataKey="i" tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: "Test sample", fill: "#64748b", fontSize: 11 }} />
                      <YAxis tickFormatter={(v) => `₹${v}`} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569" }} />
                      <Legend />
                      <Line type="monotone" dataKey="actual" name="Actual" stroke="#8b5cf6" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#ec4899" dot={false} strokeWidth={2} strokeDasharray="6 4" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </>
          )}
        </div>
      )}

      {tab === "evaluation" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <h3 className="mb-4 text-sm font-medium text-slate-300">Training & validation loss (MSE)</h3>
            {!historyChartData.length ? (
              <p className="text-slate-500">No training_history document.</p>
            ) : (
              <div style={{ width: "100%", height: 340 }}>
                <ResponsiveContainer>
                  <LineChart data={historyChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                    <XAxis dataKey="epoch" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569" }} />
                    <Legend />
                    <Line type="monotone" dataKey="train" name="Train" stroke="#8b5cf6" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="val" name="Validation" stroke="#14b8a6" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </Card>
          <Card>
            <h3 className="mb-4 text-sm font-medium text-slate-300">Ablation study (normalized MSE on test)</h3>
            {!ablationChartData.length ? (
              <p className="text-slate-500">No ablation document.</p>
            ) : (
              <div style={{ width: "100%", height: 340 }}>
                <ResponsiveContainer>
                  <BarChart data={ablationChartData} layout="vertical" margin={{ left: 24 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                    <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={120} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569" }} />
                    <Bar dataKey="mse" name="Test MSE" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
