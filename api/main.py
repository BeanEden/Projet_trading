from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from api.models import CandleInput, PredictionOutput
from api.dependencies import get_model_manager, ModelManager
from datetime import datetime

# Lifetime events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    manager = get_model_manager()
    try:
        manager.load_model()
    except Exception as e:
        print(f"Erreur au chargement du modèle: {e}")
        # On ne bloque pas l'app, mais predict retournera erreur
    yield
    # Shutdown (rien de spécial)

app = FastAPI(
    title="GBP/USD RL Trading API",
    description="API pour prédire les actions de trading (HOLD/BUY/SELL) avec un agent PPO.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Health"])
async def health_check():
    """Healthcheck endpoint."""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/status", tags=["Info"])
async def get_status(manager: ModelManager = Depends(get_model_manager)):
    """Retourne l'état du modèle et de l'historique."""
    model_loaded = manager.model is not None
    history_len = len(manager.history)
    return {
        "model_loaded": model_loaded,
        "history_length": history_len,
        "window_size_required": 20,
        "message": "Prêt à prédire" if model_loaded and history_len >= 20 else "Attente données ou modèle"
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(candle: CandleInput, manager: ModelManager = Depends(get_model_manager)):
    """
    Ajoute une bougie et retourne l'action recommandée.
    """
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    
    # 1. Ajouter la bougie
    manager.add_candle(candle)
    
    # 2. Prédire
    action_id, action_str = manager.predict()
    
    if action_id is None:
        # Pas assez de données
        raise HTTPException(status_code=400, detail=f"Données insuffisantes: {action_str}")
    
    return PredictionOutput(
        action=action_str,
        action_id=action_id,
        confidence=None,  # PPO (MlpPolicy) ne donne pas la confiance directement facilement sans accès logits
        timestamp=datetime.utcnow(),
        model_version="v1_ppo"
    )
