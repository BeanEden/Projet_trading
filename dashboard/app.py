from flask import Flask, render_template, jsonify, send_from_directory, request, session, redirect, url_for
import subprocess
import os
import sys
import json
import csv
from pathlib import Path
import time
import pandas as pd

app = Flask(__name__)
app.secret_key = "trading_bot_secret_2024"

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "v1"
DATA_DIR = BASE_DIR / "data"
EDA_DIR = BASE_DIR / "evaluation" / "eda"
BASELINE_DIR = BASE_DIR / "evaluation" / "baseline_results"
LOG_FILE = BASE_DIR / "training.log"

TRAINING_PIN = "4242"  # Code PIN pour protéger l'entraînement

training_process = None


# ── Helpers ────────────────────────────────────────────────────
def load_config():
    config_path = MODEL_DIR / "rl_config.json"
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
    for year in [2022, 2023, 2024]:
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


def get_model_info():
    model_path = MODEL_DIR / "ppo_trading.zip"
    if model_path.exists():
        stat = model_path.stat()
        return {
            "exists": True,
            "size": f"{stat.st_size / 1024 / 1024:.2f} MB",
            "date": time.strftime("%d/%m/%Y %H:%M", time.localtime(stat.st_mtime)),
        }
    return {"exists": False}


# ── Pages ──────────────────────────────────────────────────────
@app.route("/")
def index():
    config = load_config()
    model_info = get_model_info()
    data_summary = get_data_summary()
    return render_template("index.html", config=config, model=model_info, data=data_summary)


@app.route("/monitoring")
def monitoring():
    eda_images = sorted([f.name for f in EDA_DIR.glob("*.png")]) if EDA_DIR.exists() else []
    baseline_images = sorted([f.name for f in BASELINE_DIR.glob("*.png")]) if BASELINE_DIR.exists() else []
    baseline_summary = load_baseline_summary()
    return render_template(
        "monitoring.html",
        eda_images=eda_images,
        baseline_images=baseline_images,
        baseline_summary=baseline_summary,
    )


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


@app.route("/api_docs")
def api_docs():
    return render_template("api_docs.html")


# ── Fichiers statiques (images) ────────────────────────────────
@app.route("/images/eda/<path:filename>")
def eda_image(filename):
    return send_from_directory(str(EDA_DIR), filename)


@app.route("/images/baseline/<path:filename>")
def baseline_image(filename):
    return send_from_directory(str(BASELINE_DIR), filename)


# ── API Data (JSON pour Plotly) ────────────────────────────────
@app.route("/api/price_data/<int:year>")
def price_data(year):
    """Retourne les données M15 pour un graphe candlestick Plotly."""
    m15_path = DATA_DIR / "m15" / f"GBPUSD_M15_{year}.csv"
    if not m15_path.exists():
        return jsonify({"error": "Fichier non trouvé"}), 404

    df = pd.read_csv(m15_path, parse_dates=["timestamp"])
    # Sous-échantillonner à 500 points max pour performance
    step = max(1, len(df) // 500)
    df = df.iloc[::step]
    return jsonify({
        "timestamp": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "open": df["open_15m"].round(5).tolist(),
        "high": df["high_15m"].round(5).tolist(),
        "low": df["low_15m"].round(5).tolist(),
        "close": df["close_15m"].round(5).tolist(),
    })


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
        log_f = open(LOG_FILE, "w", encoding="utf-8")
        training_process = subprocess.Popen(
            [sys.executable, "training/train_rl.py"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
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
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return jsonify({"logs": f.read()})
    except Exception as e:
        return jsonify({"logs": f"Erreur: {e}"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
