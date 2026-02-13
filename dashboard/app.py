from flask import Flask, render_template, jsonify, send_from_directory, request, session, redirect, url_for
import subprocess
import os
import sys
import json
import csv
from pathlib import Path
import time
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "trading_bot_secret_2024"

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_ROOT = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
EDA_DIR = BASE_DIR / "evaluation" / "eda"
BASELINE_DIR = BASE_DIR / "evaluation" / "baseline_results"
LOG_FILE = BASE_DIR / "training.log"

TRAINING_PIN = "4242"


# ── Model Metadata (Descriptions) ──────────────────────────────
MODEL_METADATA = {
    "v1": {"method": "PPO Basic", "desc": "Modèle de base PPO sans features complexes."},
    "v2": {"method": "PPO + Indicators", "desc": "Ajout de RSI, MACD et Bollinger Bands."},
    "v3": {"method": "PPO + Patterns", "desc": "Tentative de détection de patterns (Dozi, Engulfing)."},
    "v4": {"method": "PPO + News", "desc": "Intégration expérimentale de sentiment (simulé)."},
    "v5": {"method": "PPO + Auto TP/SL", "desc": "Gestion dynamique du Take Profit et Stop Loss (Target/Stop basés sur volatilité)."},
    "v5b": {"method": "PPO + Aggif", "desc": "Variante agressive de V5."},
    "v5c": {"method": "PPO + Conservative", "desc": "Variante conservatrice de V5."},
    "v5d": {"method": "PPO + Scalping", "desc": "Stratégie court-terme (High Frequency, Low TP)."},
    "v6": {"method": "PPO + Time Features", "desc": "Ajout de l'encodage cyclique de l'heure et du jour."},
    "v6_fast": {"method": "PPO + Time (Fast)", "desc": "Entraînement rapide de V6."},
    "v7": {"method": "DQN", "desc": "Test de l'algorithme Deep Q-Network (Off-policy).Moins stable que PPO."},
    "v8": {"method": "PPO + Regime (ADX)", "desc": "Détection de régime de marché (Tendance vs Range) avec ADX et Ratio ATR."},
    "v8_fixed": {"method": "PPO + Regime + Fix", "desc": "Correction du bug 'Double Spread'. Version robuste de V8."},
    "v9": {"method": "PPO + Regime (Optimized)", "desc": "Version V8 Fixed renommée. Candidat production."},
    "v9_optuna": {"method": "PPO + Hyperopt", "desc": "Optimisation Bayésienne des hyperparamètres sur V8."},
    "v10": {"method": "PPO + 2025 Focus", "desc": "Entraînement étendu incluant 2024 pour maximiser 2025."},
}

training_process = None

# ── Cache M15 data ─────────────────────────────────────────────
_m15_cache = {}


def _load_m15(year):
    if year not in _m15_cache:
        path = DATA_DIR / "m15" / f"GBPUSD_M15_{year}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            _m15_cache[year] = df
        else:
            return None
    return _m15_cache[year]


# ── Helpers ────────────────────────────────────────────────────
def get_available_models():
    if not MODELS_ROOT.exists():
        return []
    # List subdirectories starting with 'v'
    dirs = [d.name for d in MODELS_ROOT.iterdir() if d.is_dir() and d.name.startswith("v")]
    # Sort nicely (v1, v2, ... v10)
    try:
        dirs.sort(key=lambda x: (int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0, x))
    except:
        dirs.sort()
    return dirs


def load_config(version):
    config_path = MODELS_ROOT / version / "rl_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def load_baseline_summary():
    csv_path = BASELINE_DIR / "baseline_summary.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_data_summary():
    summary = []
    for year in [2022, 2023, 2024, 2025]:
        m1 = DATA_DIR / f"DAT_MT_GBPUSD_M1_{year}.csv"
        m15 = DATA_DIR / "m15" / f"GBPUSD_M15_{year}.csv"
        entry = {"year": year}
        if m1.exists():
            entry["m1_size"] = f"{m1.stat().st_size / 1024 / 1024:.1f} MB"
        if m15.exists():
            entry["m15_size"] = f"{m15.stat().st_size / 1024 / 1024:.1f} MB"
            with open(m15, "r") as f:
                entry["m15_rows"] = sum(1 for _ in f) - 1
        summary.append(entry)
    return summary


def get_model_info(version):
    model_dir = MODELS_ROOT / version
    # Check for zip file, name could vary
    candidates = ["ppo_trading_best.zip", "dqn_trading_best.zip", "ppo_trading.zip", "dqn_trading.zip"]
    model_path = None
    for c in candidates:
        p = model_dir / c
        if p.exists():
            model_path = p
            break

    if model_path:
        stat = model_path.stat()
        return {
            "name": version,
            "exists": True,
            "file": model_path.name,
            "size": f"{stat.st_size / 1024 / 1024:.2f} MB",
            "date": time.strftime("%d/%m/%Y %H:%M", time.localtime(stat.st_mtime)),
        }
    return {"name": version, "exists": False}


def load_eval_results(version):
    path = MODELS_ROOT / version / "eval_results.json"
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
            
        # Calculate derived metrics for each year
        for year, res in data.items():
            if "win_rate" in res and "avg_win" in res and "avg_loss" in res:
                wr = res["win_rate"] / 100.0
                avg_win = res["avg_win"]
                avg_loss = abs(res["avg_loss"])
                
                # Precision = Win Rate
                res["precision"] = res["win_rate"]
                
                # Profit Factor
                if avg_loss > 0 and (1 - wr) > 0:
                    profit_factor = (wr * avg_win) / ((1 - wr) * avg_loss)
                    res["profit_factor"] = round(profit_factor, 2)
                else:
                    res["profit_factor"] = 99.99 if wr > 0.5 else 0.0

                # F1 Score (Approximate based on Trade Success)
                # Treating "Win" as Positive Class. 
                # Recall is hard to define without "Missed Trades", but we can use:
                # F1 = 2 * Precision * Recall / (Precision + Recall)
                # Assuming Recall ~ Precision (optimistic) or Recall = 1 (executed trades only)
                # Let's use F1 of the *executed trades classification*: 
                # TP / (TP + 0.5(FP + FN)). If FN=0 (we don't know missed), F1 = 2TP / (2TP + FP).
                # 2 * Win / (2 * Win + Loss)
                # 2 * wr / (2 * wr + (1-wr)) = 2*wr / (wr + 1)
                res["f1_score"] = round(2 * wr / (wr + 1) * 100, 2)

        return data
    return None


# ── Pages ──────────────────────────────────────────────────────
@app.route("/")
def index():
    version = request.args.get("version")
    models = get_available_models()
    
    if not version and models:
        # Default to the latest model
        version = models[-1]
    elif not version:
        version = "v1"

    config = load_config(version)
    model_info = get_model_info(version)
    eval_results = load_eval_results(version)
    data_summary = get_data_summary()
    return render_template("index.html", config=config, model=model_info, eval_results=eval_results, data=data_summary, models=models, current_version=version, metadata=MODEL_METADATA.get(version, {}))


@app.route("/monitoring")
def monitoring():
    baseline_summary = load_baseline_summary()
    return render_template("monitoring.html", baseline_summary=baseline_summary)


@app.route("/comparison")
def comparison():
    return render_template("comparison.html")


@app.route("/training")
def training():
    authenticated = session.get("training_auth", False)
    return render_template("training.html", authenticated=authenticated)


@app.route("/training/auth", methods=["POST"])
def training_auth():
    pin = request.form.get("pin", "")
    if pin == TRAINING_PIN:
        session["training_auth"] = True
        return redirect(url_for("training"))
    return render_template("training.html", authenticated=False, error="Code PIN incorrect.")


@app.route("/training/logout")
def training_logout():
    session.pop("training_auth", None)
    return redirect(url_for("training"))


@app.route("/api_docs")
def api_docs():
    return render_template("api_docs.html")


# ── Static images ──────────────────────────────────────────────
@app.route("/images/eda/<path:filename>")
def eda_image(filename):
    return send_from_directory(str(EDA_DIR), filename)


@app.route("/images/baseline/<path:filename>")
def baseline_image(filename):
    return send_from_directory(str(BASELINE_DIR), filename)


# ── API: Price candlestick data ────────────────────────────────
@app.route("/api/price_data/<int:year>")
def price_data(year):
    df = _load_m15(year)
    if df is None:
        return jsonify({"error": "Fichier non trouvé"}), 404
    step = max(1, len(df) // 500)
    d = df.iloc[::step]
    return jsonify({
        "timestamp": d["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "open": d["open_15m"].round(5).tolist(),
        "high": d["high_15m"].round(5).tolist(),
        "low": d["low_15m"].round(5).tolist(),
        "close": d["close_15m"].round(5).tolist(),
    })


# ── API: Returns distribution ──────────────────────────────────
@app.route("/api/returns/<int:year>")
def returns_data(year):
    df = _load_m15(year)
    if df is None:
        return jsonify({"error": "Données non trouvées"}), 404
    returns = df["close_15m"].pct_change().dropna()
    # Histogram bins
    counts, edges = np.histogram(returns, bins=80)
    centers = ((edges[:-1] + edges[1:]) / 2 * 100).tolist()  # en %
    return jsonify({
        "centers": [round(c, 4) for c in centers],
        "counts": counts.tolist(),
        "mean": round(float(returns.mean() * 100), 6),
        "std": round(float(returns.std() * 100), 4),
    })


# ── API: Rolling volatility ───────────────────────────────────
@app.route("/api/volatility/<int:year>")
def volatility_data(year):
    df = _load_m15(year)
    if df is None:
        return jsonify({"error": "Données non trouvées"}), 404
    returns = df["close_15m"].pct_change().dropna()
    vol_1h = returns.rolling(4).std() * 100  # 4 x 15min = 1h
    vol_4h = returns.rolling(16).std() * 100
    vol_1d = returns.rolling(96).std() * 100  # 96 x 15min = 24h
    step = max(1, len(df) // 400)
    ts = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").iloc[::step].tolist()
    return jsonify({
        "timestamp": ts,
        "vol_1h": vol_1h.iloc[::step].round(4).fillna(0).tolist(),
        "vol_4h": vol_4h.iloc[::step].round(4).fillna(0).tolist(),
        "vol_1d": vol_1d.iloc[::step].round(4).fillna(0).tolist(),
    })


# ── API: Hourly analysis ──────────────────────────────────────
@app.route("/api/hourly/<int:year>")
def hourly_data(year):
    df = _load_m15(year)
    if df is None:
        return jsonify({"error": "Données non trouvées"}), 404
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["return"] = df["close_15m"].pct_change()
    hourly = df.groupby("hour")["return"].agg(["mean", "std", "count"])
    return jsonify({
        "hours": list(range(24)),
        "mean_return": (hourly["mean"] * 100).round(5).tolist(),
        "volatility": (hourly["std"] * 100).round(4).tolist(),
        "count": hourly["count"].tolist(),
    })


# ── API: Metrics comparison bar chart ──────────────────────────
@app.route("/api/metrics_comparison")
def metrics_comparison():
    rows = load_baseline_summary()
    if not rows:
        return jsonify({"error": "Pas de données"}), 404
    strategies = []
    profits = []
    sharpes = []
    win_rates = []
    labels = []
    for r in rows:
        label = f"{r['Stratégie']} ({r['Année']})"
        labels.append(label)
        profits.append(float(r["Profit (%)"]))
        sharpes.append(float(r["Sharpe"]))
        win_rates.append(float(r["Win Rate (%)"]))
    return jsonify({
        "labels": labels,
        "profits": profits,
        "sharpes": sharpes,
        "win_rates": win_rates,
    })


@app.route("/api/models_comparison")
def models_comparison_api():
    models = get_available_models()
    data = {}
    
    # Structure: { "v1": { "2022": 10, "2023": -5 ... }, "v9": ... }
    for m in models:
        res = load_eval_results(m)
        if res:
            res["meta"] = MODEL_METADATA.get(m, {"method": "N/A", "desc": ""})
            data[m] = res
            
    return jsonify(data)


# ── API: Training ─────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    global training_process
    is_training = training_process is not None and training_process.poll() is None
    return jsonify({"training": is_training})


@app.route("/api/start_training", methods=["POST"])
def start_training():
    if not session.get("training_auth", False):
        return jsonify({"status": "error", "message": "Non autorisé."}), 403
    global training_process
    if training_process and training_process.poll() is None:
        return jsonify({"status": "error", "message": "Entraînement déjà en cours."}), 400
    try:
        log_f = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        # Force UTF-8 encoding in subprocess to handle emojis on Windows
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        training_process = subprocess.Popen(
            [sys.executable, "training/train_rl.py"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env=env,
        )
        return jsonify({"status": "started", "pid": training_process.pid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stop_training", methods=["POST"])
def stop_training():
    if not session.get("training_auth", False):
        return jsonify({"status": "error", "message": "Non autorisé."}), 403
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "error", "message": "Aucun entraînement en cours."}), 400


@app.route("/api/logs")
def get_logs():
    if not LOG_FILE.exists():
        return jsonify({"logs": ""})
    try:
        # Essayer utf-8 d'abord, puis cp1252 (Windows), puis latin-1 en fallback
        for enc in ["utf-8", "cp1252", "latin-1"]:
            try:
                with open(LOG_FILE, "r", encoding=enc) as f:
                    return jsonify({"logs": f.read()})
            except UnicodeDecodeError:
                continue
        # Fallback: ignorer les erreurs
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            return jsonify({"logs": f.read()})
    except Exception as e:
        return jsonify({"logs": f"Erreur: {e}"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
