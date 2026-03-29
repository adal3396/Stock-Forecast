import Dashboard from "@/components/Dashboard";

export default function Home() {
  return (
    <div className="page-wrap">
      <header className="mb-8">
        <h1 className="hero-title">Stock price forecasting</h1>
        <p className="hero-sub text-slate-400">
          STFT spectrograms + CNN — dashboard backed by MongoDB (Assignment 2)
        </p>
      </header>
      <Dashboard />
    </div>
  );
}
