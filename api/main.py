"""
FastAPI Backend — Model Registry, Training, Prediction & Data Visualization API
================================================================================
Point d'entrée unique pour le frontend Flask (programmeur + utilisateur).
Documentation auto : http://localhost:8000/docs
"""

import os
import json
import pickle
import io
import base64
import traceback
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────
#  Paths & Config
# ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "features"
REGISTRY_FILE = MODELS_DIR / "registry.json"

AVAILABLE_FEATURES = [
    "return_1", "return_4", "ema_20", "ema_50", "ema_diff",
    "rsi_14", "rolling_std_20", "body", "ema_200",
    "distance_to_ema200", "atr_14", "adx_14",
]

ALGO_MAP = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
}

# ────────────────────────────────────────────
#  FastAPI App
# ────────────────────────────────────────────
app = FastAPI(
    title="Trading Model Registry API",
    description="API pour le versioning, l'entraînement et la prédiction de modèles ML de trading GBP/USD.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────
def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"models": {}, "published": []}


def _save_registry(reg: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=4)


_DATA_CACHE = None

def _load_data():
    """Charge et concatène les 3 fichiers CSV de features (avec cache mémoire)."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE.copy()
    frames = []
    for year in [2022, 2023, 2024]:
        p = DATA_DIR / f"GBPUSD_M15_{year}_features.csv"
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp"))
    if not frames:
        raise FileNotFoundError("Aucun fichier de données trouvé dans data/features/")
    df = pd.concat(frames).sort_index()
    df["target_return"] = df["close_15m"].shift(-1) - df["close_15m"]
    df["target"] = (df["target_return"] > 0).astype(int)
    df.dropna(inplace=True)
    _DATA_CACHE = df
    return _DATA_CACHE.copy()


def _fig_to_b64(fig) -> str:
    """Convertit un matplotlib Figure en data URI base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _load_model_from_registry(name: str, version: str):
    """Charge un modèle pickle depuis le registry."""
    reg = _load_registry()
    if name not in reg["models"]:
        raise HTTPException(404, f"Modèle '{name}' inconnu")
    entries = reg["models"][name]
    if version == "latest":
        meta = entries[-1]
    else:
        meta = next((m for m in entries if m["version"] == version), None)
        if not meta:
            raise HTTPException(404, f"Version '{version}' introuvable pour '{name}'")
    pkl_path = MODELS_DIR / name / meta["version"] / "model.pkl"
    if not pkl_path.exists():
        raise HTTPException(404, f"Fichier modèle introuvable: {pkl_path}")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    return model, meta


# ────────────────────────────────────────────
#  Pydantic Schemas
# ────────────────────────────────────────────
class TrainRequest(BaseModel):
    model_name: str = "rf_direction_classifier"
    algorithm: str = "RandomForest"
    features: list[str] = ["rsi_14", "ema_20", "ema_50", "atr_14", "adx_14"]
    params: dict = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    author: str = "JCLoirat"
    grid_search: bool = False


class PublishRequest(BaseModel):
    model_name: str
    version: str


# ════════════════════════════════════════════
#  ROUTES : DATA
# ════════════════════════════════════════════
@app.get("/data/columns")
def data_columns():
    """Retourne les colonnes disponibles dans les features."""
    return {"columns": AVAILABLE_FEATURES}


@app.get("/data/summary")
def data_summary():
    """Résumé statistique des données par année."""
    df = _load_data()
    result = {}
    for year in [2022, 2023, 2024]:
        sub = df.loc[str(year)]
        result[str(year)] = {
            "nb_rows": len(sub),
            "close_mean": round(float(sub["close_15m"].mean()), 5),
            "close_std": round(float(sub["close_15m"].std()), 5),
            "target_balance": round(float(sub["target"].mean()), 4),
        }
    return result


@app.get("/data/timeseries")
def data_timeseries(year: int = 2024, column: str = "close_15m", resample: str = "1D"):
    """Retourne une série temporelle resampleée."""
    df = _load_data()
    sub = df.loc[str(year)]
    if column not in sub.columns:
        raise HTTPException(400, f"Colonne '{column}' inconnue")
    ts = sub[column].resample(resample).mean().dropna()
    return {"dates": ts.index.strftime("%Y-%m-%d").tolist(), "values": ts.values.tolist()}


# ════════════════════════════════════════════
#  ROUTES : REGISTRY
# ════════════════════════════════════════════
@app.get("/registry/models")
def registry_list():
    """Liste tous les modèles et leurs versions."""
    reg = _load_registry()
    return reg["models"]


@app.get("/registry/history/{model_name}")
def registry_history(model_name: str):
    """Historique complet d'un modèle."""
    reg = _load_registry()
    if model_name not in reg["models"]:
        raise HTTPException(404, f"Modèle '{model_name}' inconnu")
    return reg["models"][model_name]


@app.get("/registry/published")
def registry_published():
    """Liste les modèles publiés pour l'utilisateur final."""
    reg = _load_registry()
    return reg.get("published", [])


@app.post("/registry/publish")
def registry_publish(req: PublishRequest):
    """Publie un modèle/version pour l'interface utilisateur."""
    reg = _load_registry()
    if req.model_name not in reg["models"]:
        raise HTTPException(404, f"Modèle '{req.model_name}' inconnu")
    entry = next((m for m in reg["models"][req.model_name] if m["version"] == req.version), None)
    if not entry:
        raise HTTPException(404, f"Version '{req.version}' introuvable")
    pub_entry = {"model_name": req.model_name, "version": req.version, "published_at": datetime.now().isoformat()}
    if "published" not in reg:
        reg["published"] = []
    # Remove previous publication of same model
    reg["published"] = [p for p in reg["published"] if p["model_name"] != req.model_name or p["version"] != req.version]
    reg["published"].append(pub_entry)
    _save_registry(reg)
    return {"status": "ok", "message": f"{req.model_name} {req.version} publié"}


@app.post("/registry/unpublish")
def registry_unpublish(req: PublishRequest):
    """Retire un modèle publié."""
    reg = _load_registry()
    reg["published"] = [p for p in reg.get("published", []) if not (p["model_name"] == req.model_name and p["version"] == req.version)]
    _save_registry(reg)
    return {"status": "ok"}


@app.delete("/registry/model/{model_name}/{version}")
def delete_model_version(model_name: str, version: str):
    """Supprime une version spécifique d'un modèle (fichiers + entrée registry)."""
    reg = _load_registry()
    if model_name not in reg["models"]:
        raise HTTPException(404, "Modèle introuvable")
    
    # Filter out the specific version
    original_len = len(reg["models"][model_name])
    reg["models"][model_name] = [m for m in reg["models"][model_name] if m["version"] != version]
    
    if len(reg["models"][model_name]) == original_len:
         raise HTTPException(404, "Version introuvable")
         
    # Update published list if necessary
    if "published" in reg:
        reg["published"] = [p for p in reg.get("published", []) if not (p["model_name"] == model_name and p["version"] == version)]

    # Delete files
    version_dir = MODELS_DIR / model_name / version
    if version_dir.exists():
        shutil.rmtree(version_dir)
        
    # If no versions left, remove model key and dir? Optional.
    # Let's keep the model key for now or remove if empty.
    if not reg["models"][model_name]:
        del reg["models"][model_name]
        shutil.rmtree(MODELS_DIR / model_name, ignore_errors=True)
        
    _save_registry(reg)
    return {"status": "ok", "message": f"Version {version} deleted"}


# ════════════════════════════════════════════
#  ROUTES : TRAINING
# ════════════════════════════════════════════
@app.post("/train")
def train_model(req: TrainRequest):
    """Entraîne un nouveau modèle et l'enregistre dans le registry."""
    try:
        df = _load_data()
        train_df = df.loc["2022"]
        test_df = df.loc["2024"]

        feats = [f for f in req.features if f in AVAILABLE_FEATURES]
        if not feats:
            raise HTTPException(400, "Aucune feature valide fournie")

        X_train, y_train = train_df[feats], train_df["target"]
        X_test, y_test = test_df[feats], test_df["target"]

        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Algo
        algo_cls = ALGO_MAP.get(req.algorithm)
        if not algo_cls:
            raise HTTPException(400, f"Algorithme inconnu: {req.algorithm}")

        model = algo_cls(**req.params)
        
        best_params = req.params
        grid_results = None

        if req.grid_search:
            # Param Grid Defaults
            param_grid = {}
            if req.algorithm == "RandomForest":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5],
                }
            elif req.algorithm == "GradientBoosting":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5],
                }
            elif req.algorithm == "LogisticRegression":
                param_grid = {
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                }
            
            # Grid Search
            print(f"Starting GridSearchCV for {req.algorithm}...")
            grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=1)
            grid.fit(X_train_s, y_train)
            print("GridSearchCV Done.")
            
            model = grid.best_estimator_
            best_params = grid.best_params_
            grid_results = {
                "best_score": round(grid.best_score_, 4),
                "best_params": best_params
            }
        else:     
            model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None

        metrics = {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}
        if auc is not None:
            metrics["roc_auc"] = round(auc, 4)

        # Save model + scaler
        reg = _load_registry()
        if req.model_name not in reg["models"]:
            reg["models"][req.model_name] = []
        version_id = len(reg["models"][req.model_name]) + 1
        version_tag = f"v{version_id}"

        model_dir = MODELS_DIR / req.model_name / version_tag
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        meta = {
            "version": version_tag,
            "timestamp": datetime.now().isoformat(),
            "author": req.author,
            "algorithm": req.algorithm,
            "features": feats,
            "params": best_params,
            "metrics": metrics,
            "grid_search": grid_results,
        }
        with open(model_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)

        reg["models"][req.model_name].append(meta)
        _save_registry(reg)

        return {"status": "ok", "version": version_tag, "metrics": metrics, "grid_search": grid_results, "best_params": best_params}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur entraînement: {traceback.format_exc()}")


# ════════════════════════════════════════════
#  ROUTES : EVALUATION / CHARTS
# ════════════════════════════════════════════
@app.get("/evaluate/{model_name}/{version}")
def evaluate_model(model_name: str, version: str):
    """Évaluation complète d'un modèle : classification report, feature importance, etc."""
    model, meta = _load_model_from_registry(model_name, version)
    df = _load_data()
    test_df = df.loc["2024"]

    feats = meta.get("features", AVAILABLE_FEATURES[:5])
    feats = [f for f in feats if f in test_df.columns]

    # Check if scaler exists
    scaler_path = MODELS_DIR / model_name / meta["version"] / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test = scaler.transform(test_df[feats])
    else:
        X_test = test_df[feats].values

    y_test = test_df["target"]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    # Qualitative assessment
    if acc > 0.6:
        assessment = "🟢 Performance acceptable — le modèle surpasse le hasard."
    elif acc > 0.52:
        assessment = "🟡 Performance marginale — le modèle est légèrement meilleur que le hasard (50%). Attention à l'overfit."
    else:
        assessment = "🔴 Performance faible — le modèle est quasi-aléatoire (≈50%). Non exploitable en production."

    result = {
        "classification_report": report,
        "accuracy": round(acc, 4),
        "assessment": assessment,
        "meta": meta,
    }

    # AUC
    if y_proba is not None:
        result["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)

    return result


@app.get("/charts/confusion/{model_name}/{version}")
def chart_confusion(model_name: str, version: str):
    """Matrice de confusion en base64."""
    model, meta = _load_model_from_registry(model_name, version)
    df = _load_data()
    test_df = df.loc["2024"]
    feats = meta.get("features", AVAILABLE_FEATURES[:5])
    feats = [f for f in feats if f in test_df.columns]

    scaler_path = MODELS_DIR / model_name / meta["version"] / "scaler.pkl"
    X = test_df[feats].values
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)

    y_pred = model.predict(X)
    cm = confusion_matrix(test_df["target"], y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", ax=ax,
                xticklabels=["Baisse", "Hausse"], yticklabels=["Baisse", "Hausse"])
    ax.set_title(f"Matrice de Confusion — {model_name} {meta['version']}")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    return {"image": _fig_to_b64(fig)}


@app.get("/charts/roc/{model_name}/{version}")
def chart_roc(model_name: str, version: str):
    """Courbe ROC-AUC en base64."""
    model, meta = _load_model_from_registry(model_name, version)
    df = _load_data()
    test_df = df.loc["2024"]
    feats = meta.get("features", AVAILABLE_FEATURES[:5])
    feats = [f for f in feats if f in test_df.columns]

    scaler_path = MODELS_DIR / model_name / meta["version"] / "scaler.pkl"
    X = test_df[feats].values
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)

    if not hasattr(model, "predict_proba"):
        return {"image": None, "error": "Le modèle ne supporte pas predict_proba"}

    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(test_df["target"], y_proba)
    auc_val = roc_auc_score(test_df["target"], y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#4E8D7C", lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (0.500)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#4E8D7C")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Courbe ROC — {model_name} {meta['version']}")
    ax.legend()
    return {"image": _fig_to_b64(fig)}


@app.get("/charts/feature_importance/{model_name}/{version}")
def chart_feature_importance(model_name: str, version: str):
    """Feature importance pour les modèles tree-based."""
    model, meta = _load_model_from_registry(model_name, version)
    if not hasattr(model, "feature_importances_"):
        return {"image": None, "error": "Ce modèle n'a pas de feature_importances_"}

    feats = meta.get("features", AVAILABLE_FEATURES[:5])
    importances = model.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([feats[i] for i in order], importances[order], color="#D8C3A5")
    ax.set_title(f"Feature Importance — {model_name} {meta['version']}")
    ax.set_xlabel("Importance")
    return {"image": _fig_to_b64(fig)}


@app.get("/charts/clusters")
def chart_clusters(n_clusters: int = 3, features: str = "rsi_14,ema_diff,atr_14"):
    """KMeans clustering + silhouette score sur les données 2024."""
    df = _load_data()
    test_df = df.loc["2024"]
    feat_list = [f.strip() for f in features.split(",") if f.strip() in test_df.columns]
    if len(feat_list) < 2:
        raise HTTPException(400, "Il faut au moins 2 features pour le clustering")

    X = test_df[feat_list].dropna().values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_s)
    sil = silhouette_score(X_s, labels)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_s)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="Set2", alpha=0.4, s=8)
    ax.set_title(f"KMeans Clusters (k={n_clusters}, Silhouette={sil:.3f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    return {"image": _fig_to_b64(fig), "silhouette": round(sil, 4)}


@app.get("/charts/anova")
def chart_anova(feature: str = "rsi_14"):
    """ANOVA entre la feature et la target (Hausse vs Baisse)."""
    df = _load_data()
    test_df = df.loc["2024"]
    if feature not in test_df.columns:
        raise HTTPException(400, f"Feature '{feature}' inconnue")

    group_0 = test_df[test_df["target"] == 0][feature].dropna()
    group_1 = test_df[test_df["target"] == 1][feature].dropna()
    f_stat, p_val = f_oneway(group_0, group_1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([group_0.values, group_1.values], labels=["Baisse (0)", "Hausse (1)"])
    ax.set_title(f"ANOVA — {feature}\nF={f_stat:.2f}, p={p_val:.4f}")
    ax.set_ylabel(feature)

    assessment = "🟢 Significatif" if p_val < 0.05 else "🔴 Non significatif"

    return {"image": _fig_to_b64(fig), "f_statistic": round(f_stat, 4), "p_value": round(p_val, 6), "assessment": assessment}


# ════════════════════════════════════════════
#  ROUTES : PREDICTION (pour l'utilisateur)
# ════════════════════════════════════════════
@app.get("/predict/{model_name}/{version}")
def predict(model_name: str, version: str, granularity: str = "1D"):
    """
    Prédiction vs Réel sur 2024.
    granularity: '1D' (jour), '1W' (semaine), '1M' (mois).
    Montre l'accuracy par période pour illustrer l'efficacité court-terme.
    """
    model, meta = _load_model_from_registry(model_name, version)
    df = _load_data()
    test_df = df.loc["2024"].copy()
    feats = meta.get("features", AVAILABLE_FEATURES[:5])
    feats = [f for f in feats if f in test_df.columns]

    scaler_path = MODELS_DIR / model_name / meta["version"] / "scaler.pkl"
    X = test_df[feats].values
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)

    test_df["prediction"] = model.predict(X)

    # Accuracy globale
    global_acc = accuracy_score(test_df["target"], test_df["prediction"])

    # Accuracy par période
    test_df["period"] = test_df.index.to_period(granularity)
    period_acc = test_df.groupby("period").apply(
        lambda g: accuracy_score(g["target"], g["prediction"]) if len(g) > 0 else 0
    )

    # Chart
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})

    # Top: Price + correct/incorrect predictions
    correct = test_df["prediction"] == test_df["target"]
    axes[0].plot(test_df.index, test_df["close_15m"], color="#8E8D8A", alpha=0.5, lw=0.5, label="Prix GBP/USD")
    axes[0].scatter(test_df.index[correct], test_df["close_15m"][correct], c="#4E8D7C", s=1, alpha=0.3, label="Correct")
    axes[0].scatter(test_df.index[~correct], test_df["close_15m"][~correct], c="#E85A4F", s=1, alpha=0.3, label="Incorrect")
    axes[0].set_title(f"Prédictions vs Réel 2024 — {model_name} {meta['version']} (Acc={global_acc:.1%})")
    axes[0].legend(markerscale=5)

    # Bottom: Accuracy par période
    gran_label = {"1D": "Jour", "1W": "Semaine", "1ME": "Mois"}.get(granularity, granularity)
    colors = ["#4E8D7C" if v > 0.52 else "#E85A4F" for v in period_acc.values]
    axes[1].bar(range(len(period_acc)), period_acc.values, color=colors, alpha=0.7)
    axes[1].axhline(0.5, color="red", ls="--", alpha=0.5, label="Seuil 50%")
    axes[1].set_ylabel(f"Accuracy / {gran_label}")
    axes[1].set_title(f"Accuracy par {gran_label}")
    axes[1].legend()
    fig.tight_layout()

    return {
        "image": _fig_to_b64(fig),
        "global_accuracy": round(global_acc, 4),
        "period_accuracy": {str(k): round(v, 4) for k, v in period_acc.items()},
        "nb_periods": len(period_acc),
        "meta": meta,
    }


@app.get("/compare")
def compare_models(names: str = "", versions: str = ""):
    """
    Compare plusieurs modèles/versions.
    names: 'rf_direction_classifier,rf_direction_classifier'
    versions: 'v1,v2'
    """
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    ver_list = [v.strip() for v in versions.split(",") if v.strip()]

    if len(name_list) != len(ver_list) or len(name_list) < 2:
        raise HTTPException(400, "Fournir au moins 2 paires name,version (séparées par des virgules)")

    df = _load_data()
    test_df = df.loc["2024"]
    comparison = []

    for name, ver in zip(name_list, ver_list):
        model, meta = _load_model_from_registry(name, ver)
        feats = meta.get("features", AVAILABLE_FEATURES[:5])
        feats = [f for f in feats if f in test_df.columns]

        scaler_path = MODELS_DIR / name / meta["version"] / "scaler.pkl"
        X = test_df[feats].values
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)

        y_pred = model.predict(X)
        acc = accuracy_score(test_df["target"], y_pred)
        f1 = f1_score(test_df["target"], y_pred)

        row = {
            "model_name": name,
            "version": ver,
            "algorithm": meta.get("algorithm", "Unknown"),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "features": feats,
            "params": meta.get("params", {}),
            "timestamp": meta.get("timestamp", ""),
        }
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
            row["roc_auc"] = round(float(roc_auc_score(test_df["target"], y_proba)), 4)
        comparison.append(row)

    return {"comparison": comparison}


# ────────────────────────────────────────────
#  Launch
# ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
