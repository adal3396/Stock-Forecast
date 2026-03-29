"use client";

import { useCallback, useEffect, useRef } from "react";

function infernoColor(t) {
  const x = Math.max(0, Math.min(1, t));
  const stops = [
    [0, 0, 3],
    [28, 16, 68],
    [79, 18, 123],
    [124, 31, 108],
    [176, 46, 85],
    [223, 82, 57],
    [243, 151, 41],
    [252, 253, 191],
  ];
  const n = stops.length - 1;
  const i = x * n;
  const j = Math.floor(i);
  const f = i - j;
  const a = stops[j];
  const b = stops[Math.min(j + 1, n)];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

/** power_db: rows = frequency bins, cols = time frames */
export default function SpectrogramHeatmap({ power_db, width = 640, height = 280 }) {
  const canvasRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !power_db?.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const F = power_db.length;
    const T = power_db[0]?.length ?? 0;
    if (T === 0) return;

    let vmin = Infinity;
    let vmax = -Infinity;
    for (let f = 0; f < F; f++) {
      for (let t = 0; t < T; t++) {
        const v = power_db[f][t];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    const span = vmax - vmin || 1;

    const cw = canvas.width;
    const ch = canvas.height;
    const cellW = cw / T;
    const cellH = ch / F;

    for (let f = 0; f < F; f++) {
      for (let t = 0; t < T; t++) {
        const norm = (power_db[f][t] - vmin) / span;
        ctx.fillStyle = infernoColor(norm);
        ctx.fillRect(t * cellW, (F - 1 - f) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [power_db]);

  useEffect(() => {
    draw();
  }, [draw]);

  if (!power_db?.length) return null;

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ width: "100%", maxWidth: width, height: "auto", borderRadius: 8 }}
      aria-label="STFT spectrogram heatmap"
    />
  );
}
