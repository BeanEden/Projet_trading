import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from api.models import CandleInput

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "v1" / "ppo_trading.zip"
DATA_DIR = PROJECT_ROOT / "data" / "m15"
WINDOW_SIZE = 20

class ModelManager:
    def __init__(self):
        self.model = None
        self.history = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.last_prediction = None

    def load_model(self):
        """Charge le modèle PPO et initialise l'historique."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable à : {MODEL_PATH}")
        
        print(f"Chargement du modèle depuis {MODEL_PATH}...")
        self.model = PPO.load(MODEL_PATH)
        print("Modèle chargé avec succès.")

        # Charger l'historique initial (dernières bougies 2024 ou 2023)
        self._init_history()

    def _init_history(self):
        """Initialise l'historique avec les dernières données disponibles."""
        try:
            # Essayer de charger 2024, sinon 2023
            for year in [2024, 2023, 2022]:
                file_path = DATA_DIR / f"GBPUSD_M15_{year}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
                    # Garder les dernières N bougies
                    last_rows = df.iloc[-WINDOW_SIZE:].copy()
                    # Renommer les colonnes pour correspondre à CandleInput
                    # Le CSV a: open_15m, high_15m, low_15m, close_15m, volume_15m
                    rename_map = {
                        "open_15m": "open", "high_15m": "high", 
                        "low_15m": "low", "close_15m": "close", 
                        "volume_15m": "volume"
                    }
                    last_rows = last_rows.rename(columns=rename_map)
                    self.history = last_rows[["open", "high", "low", "close", "volume"]]
                    print(f"Historique initialisé avec {len(self.history)} bougies de {year}")
                    return
        except Exception as e:
            print(f"Erreur lors de l'initialisation de l'historique : {e}")
            print("Démarrage avec historique vide.")

    def add_candle(self, candle: CandleInput):
        """Ajoute une bougie à l'historique."""
        new_row = pd.DataFrame([{
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume
        }], index=[pd.Timestamp(candle.timestamp)])
        
        self.history = pd.concat([self.history, new_row])
        # Garder seulement la taille nécessaire (pour fenêtre + calculs)
        # On garde un peu plus que WINDOW_SIZE pour les calculs d'indicateurs (EMA, RSI nécessite plus)
        # EMA(50) a besoin de 50 périodes min.
        keep_size = 100 
        if len(self.history) > keep_size:
            self.history = self.history.iloc[-keep_size:]

    def predict(self):
        """Effectue une prédiction si assez de données."""
        if len(self.history) < WINDOW_SIZE + 50: # Besoin de plus d'historique pour EMA50
            return None, f"Pas assez de données (besoin de {WINDOW_SIZE+50} bougies)"

        df = self.history.copy().reset_index(drop=True)
        
        # ── Feature Engineering (Doit matcher EXACTEMENT TradingEnv) ──
        # Colonnes: open, high, low, close, volume (déjà renommées)
        # Mais TradingEnv utilise open_15m, high_15m... ici on a open, high...
        # On renomme temporairement pour matcher la logique si besoin, ou on adapte le code.
        # On adapte le code pour utiliser open, high, low, close.
        
        # Return court terme
        df["return_1"] = df["close"].pct_change(1)
        df["return_4"] = df["close"].pct_change(4)

        # EMA
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_diff"] = (df["ema_20"] - df["ema_50"]) / df["close"]

        # RSI 14
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"] / 100  # normaliser [0,1]

        # Volatilité
        df["rolling_std_20"] = df["return_1"].rolling(20).std()
        df["range_15m"] = (df["high"] - df["low"]) / df["close"]

        # Structure bougie
        df["body"] = (df["close"] - df["open"]) / df["close"]
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

        feature_cols = [
            "return_1", "return_4", "ema_diff", "rsi_14",
            "rolling_std_20", "range_15m", "body", "upper_wick", "lower_wick"
        ]
        
        # Remplir les NaN (comme dans l'env)
        for col in feature_cols:
            df[col] = df[col].fillna(0)

        # Prendre la fenêtre de features
        # L'index -1 est la bougie qu'on vient d'ajouter (current step)
        # On a besoin des WINDOW_SIZE dernières lignes
        # Attention: dans TradingEnv, get_observation prend [start:end]
        
        window_data = df.iloc[-WINDOW_SIZE:][feature_cols]
        
        # Normalisation Z-score locale
        mean = window_data.mean()
        std = window_data.std().replace(0, 1e-8)
        norm_data = (window_data - mean) / std
        
        # Flatten + Position
        obs = norm_data.values.flatten().astype(np.float32)
        
        # Ajouter position (supposée neutre 0)
        current_position = 0.0 
        obs = np.append(obs, current_position)
        
        # Remplacer NaN par 0 en sécurité
        obs = np.nan_to_num(obs)
        
        # Prédiction
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Map action ID to string
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return int(action), action_map.get(int(action), "UNKNOWN")


# Instance unique (Singleton)
manager = ModelManager()

def get_model_manager():
    return manager
