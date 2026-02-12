from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_model_manager, ModelManager
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

def test_status_no_model():
    # Par défaut, le modèle n'est pas chargé dans le test sans mock
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "history_length" in data

def test_predict_no_model():
    # Sans modèle, doit retourner 503
    payload = {
        "timestamp": "2024-01-01T12:00:00",
        "open": 1.2, "high": 1.3, "low": 1.1, "close": 1.25, "volume": 1000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Modèle non chargé."

# On pourrait mocker le ModelManager pour tester predict avec succès,
# mais cela demande de mocker self.model + self.history.
# Pour l'instant, ces tests suffisent pour valider la structure.
