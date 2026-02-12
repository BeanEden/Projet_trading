"""
Flask Frontend — Interface Programmeur + Utilisateur
=====================================================
Communique avec le backend FastAPI (port 8000) via requests.
"""

import os
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "trading-model-registry-2026"

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")


def api_get(endpoint, params=None):
    """Helper GET vers FastAPI."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ API FastAPI non accessible. Lancez : uvicorn api.main:app --reload --port 8000"}
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint, json_data=None):
    """Helper POST vers FastAPI."""
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ API FastAPI non accessible."}
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════
#  PROGRAMMER ROUTES
# ════════════════════════════════════════════════════
@app.route("/programmer")
def programmer_dashboard():
    models = api_get("/registry/models")
    summary = api_get("/data/summary")
    published = api_get("/registry/published")
    return render_template("programmer/dashboard.html",
                         models=models, summary=summary, published=published)


@app.route("/programmer/model/<model_name>")
def programmer_model_detail(model_name):
    history = api_get(f"/registry/history/{model_name}")
    columns = api_get("/data/columns")
    return render_template("programmer/model_detail.html",
                         model_name=model_name, history=history,
                         columns=columns.get("columns", []))


@app.route("/programmer/evaluate/<model_name>/<version>")
def programmer_evaluate(model_name, version):
    evaluation = api_get(f"/evaluate/{model_name}/{version}")
    confusion = api_get(f"/charts/confusion/{model_name}/{version}")
    roc = api_get(f"/charts/roc/{model_name}/{version}")
    importance = api_get(f"/charts/feature_importance/{model_name}/{version}")
    return render_template("programmer/evaluate.html",
                         model_name=model_name, version=version,
                         evaluation=evaluation, confusion=confusion,
                         roc=roc, importance=importance)


@app.route("/programmer/compare")
def programmer_compare():
    models = api_get("/registry/models")
    # Get all model_name + version pairs
    model_versions = []
    if isinstance(models, dict) and "error" not in models:
        for name, versions in models.items():
            for v in versions:
                model_versions.append({"name": name, "version": v["version"]})

    comparison = None
    sel_a = request.args.get("sel_a")
    sel_b = request.args.get("sel_b")
    if sel_a and sel_b:
        na, va = sel_a.split("|")
        nb, vb = sel_b.split("|")
        comparison = api_get("/compare", params={"names": f"{na},{nb}", "versions": f"{va},{vb}"})

    return render_template("programmer/compare.html",
                         model_versions=model_versions, comparison=comparison,
                         sel_a=sel_a, sel_b=sel_b)


@app.route("/programmer/train", methods=["GET", "POST"])
def programmer_train():
    columns = api_get("/data/columns")
    result = None

    if request.method == "POST":
        algo = request.form.get("algorithm", "RandomForest")
        features = request.form.getlist("features")
        model_name = request.form.get("model_name", "rf_direction_classifier")
        author = request.form.get("author", "JCLoirat")

        params = {}
        if algo == "RandomForest":
            params = {
                "n_estimators": int(request.form.get("n_estimators", 100)),
                "max_depth": int(request.form.get("max_depth", 5)),
                "random_state": 42,
            }
        elif algo == "GradientBoosting":
            params = {
                "n_estimators": int(request.form.get("n_estimators", 100)),
                "max_depth": int(request.form.get("max_depth", 3)),
                "learning_rate": float(request.form.get("learning_rate", 0.1)),
                "random_state": 42,
            }
        elif algo == "LogisticRegression":
            params = {
                "C": float(request.form.get("C", 1.0)),
                "max_iter": 1000,
                "random_state": 42,
            }

        result = api_post("/train", {
            "model_name": model_name,
            "algorithm": algo,
            "features": features,
            "params": params,
            "author": author,
        })

    return render_template("programmer/train.html",
                         columns=columns.get("columns", []),
                         result=result)


@app.route("/programmer/features")
def programmer_features():
    columns = api_get("/data/columns")
    anova_feat = request.args.get("feature", "rsi_14")
    anova = api_get("/charts/anova", params={"feature": anova_feat})
    clusters = api_get("/charts/clusters")
    return render_template("programmer/features.html",
                         columns=columns.get("columns", []),
                         anova=anova, anova_feat=anova_feat,
                         clusters=clusters)


@app.route("/programmer/publish", methods=["POST"])
def programmer_publish():
    model_name = request.form.get("model_name")
    version = request.form.get("version")
    api_post("/registry/publish", {"model_name": model_name, "version": version})
    flash(f"✅ {model_name} {version} publié pour l'utilisateur final.", "success")
    return redirect(url_for("programmer_model_detail", model_name=model_name))


@app.route("/programmer/unpublish", methods=["POST"])
def programmer_unpublish():
    model_name = request.form.get("model_name")
    version = request.form.get("version")
    api_post("/registry/unpublish", {"model_name": model_name, "version": version})
    flash(f"❎ {model_name} {version} retiré de la publication.", "info")
    return redirect(url_for("programmer_dashboard"))


# ════════════════════════════════════════════════════
#  USER ROUTES
# ════════════════════════════════════════════════════
@app.route("/")
def index():
    return redirect(url_for("user_dashboard"))


@app.route("/user")
def user_dashboard():
    published = api_get("/registry/published")
    summary = api_get("/data/summary")
    ts = api_get("/data/timeseries", params={"year": 2024, "column": "close_15m", "resample": "1D"})
    return render_template("user/dashboard.html",
                         published=published, summary=summary, timeseries=ts)


@app.route("/user/predict/<model_name>/<version>")
def user_predict(model_name, version):
    granularity = request.args.get("granularity", "1D")
    prediction = api_get(f"/predict/{model_name}/{version}", params={"granularity": granularity})
    evaluation = api_get(f"/evaluate/{model_name}/{version}")
    return render_template("user/predict.html",
                         model_name=model_name, version=version,
                         prediction=prediction, evaluation=evaluation,
                         granularity=granularity)


@app.route("/user/compare")
def user_compare():
    published = api_get("/registry/published")
    comparison = None
    if len(published) >= 2 if isinstance(published, list) else False:
        names = ",".join([p["model_name"] for p in published])
        versions = ",".join([p["version"] for p in published])
        comparison = api_get("/compare", params={"names": names, "versions": versions})
    return render_template("user/compare.html",
                         published=published, comparison=comparison)


# ────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
