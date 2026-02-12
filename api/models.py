from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class CandleInput(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp de la bougie (UTC)")
    open: float = Field(..., gt=0, description="Prix d'ouverture")
    high: float = Field(..., gt=0, description="Prix haut")
    low: float = Field(..., gt=0, description="Prix bas")
    close: float = Field(..., gt=0, description="Prix de fermeture")
    volume: float = Field(..., ge=0, description="Volume")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-01T12:00:00",
                "open": 1.2750,
                "high": 1.2765,
                "low": 1.2740,
                "close": 1.2760,
                "volume": 1250.0
            }
        }

class PredictionOutput(BaseModel):
    action: str = Field(..., description="Action recommandée : HOLD, BUY, SELL")
    action_id: int = Field(..., description="ID de l'action (0, 1, 2)")
    confidence: Optional[float] = Field(None, description="Confiance de la prédiction (si disponible)")
    timestamp: datetime = Field(..., description="Timestamp de la prédiction")
    model_version: str = Field(..., description="Version du modèle utilisé")
