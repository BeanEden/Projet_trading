from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from pathlib import Path
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

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_root():
    """Interface Web simple."""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>GBP/USD Trading API</title>
            <meta charset="utf-8">
            <style>
                body { font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f2f5; color: #333; }
                .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
                h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
                h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
                .status-item { margin: 10px 0; font-size: 1.1em; }
                .status-ok { color: #27ae60; font-weight: bold; }
                .status-err { color: #c0392b; font-weight: bold; }
                button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 1em; transition: background 0.2s; }
                button:hover { background: #2980b9; }
                ul { list-style-type: none; padding: 0; }
                li { background: #f8f9fa; padding: 10px; margin-bottom: 5px; border-radius: 4px; border-left: 4px solid #3498db; }
            </style>
        </head>
        <body>
            <h1>GBP/USD Trading API 🤖</h1>
            
            <div class="card">
                <h2>État du système</h2>
                <div id="status-display">
                    <p>Chargement du statut...</p>
                </div>
                <div style="text-align: right; margin-top: 15px;">
                    <button onclick="fetchStatus()">Actualiser Statut</button>
                </div>
            </div>

            <div class="card">
                <h2>Modèles disponibles (models/v1/)</h2>
                <ul id="models-list">Chargement...</ul>
                <div style="text-align: right; margin-top: 15px;">
                    <button onclick="fetchModels()">Actualiser Modèles</button>
                </div>
            </div>

            <div class="card">
                <h2>Documentation</h2>
                <p>Accéder à la documentation interactive de l'API (Swagger UI) :</p>
                <a href="/docs" style="display: inline-block; background: #27ae60; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px;">Ouvrir /docs</a>
            </div>

            <script>
                async function fetchStatus() {
                    const div = document.getElementById('status-display');
                    try {
                        const res = await fetch('/status');
                        const data = await res.json();
                        div.innerHTML = `
                            <div class="status-item">Modèle chargé : <span class="${data.model_loaded ? 'status-ok' : 'status-err'}">${data.model_loaded ? 'OUI' : 'NON'}</span></div>
                            <div class="status-item">Historique : <b>${data.history_length}</b> bougies (Requis: ${data.window_size_required})</div>
                            <div class="status-item">Message : <i>${data.message}</i></div>
                        `;
                    } catch (e) {
                        div.innerHTML = `<span class="status-err">Erreur de connexion API (Serveur éteint ?)</span>`;
                    }
                }

                async function fetchModels() {
                    const ul = document.getElementById('models-list');
                    try {
                        const res = await fetch('/models');
                        const data = await res.json();
                        if (data.models.length === 0) {
                            ul.innerHTML = '<li>Aucun modèle trouvé dans models/v1/</li>';
                        } else {
                            ul.innerHTML = data.models.map(m => `<li>📄 ${m}</li>`).join('');
                        }
                    } catch (e) {
                        ul.innerHTML = '<li>Erreur récupération modèles</li>';
                    }
                }

                // Initial load
                fetchStatus();
                fetchModels();
                
                // Refresh status every 5s
                setInterval(fetchStatus, 5000);
            </script>
        </body>
    </html>
    """

@app.get("/models", tags=["Info"])
async def list_models():
    """Liste les fichiers modèles .zip dans models/v1/."""
    model_dir = Path("models/v1")
    if not model_dir.exists():
        return {"models": []}
    
    files = [f.name for f in model_dir.glob("*.zip")]
    return {"models": files}

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
