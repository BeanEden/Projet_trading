# Conception RL – Système de trading GBP/USD

## 1. Problème métier
- **Objectif** : Maximiser le profit cumulé sur GBP/USD M15 en décidant BUY/SELL/HOLD à chaque bougie
- **Contraintes** : Coûts de transaction (~2 pips), drawdown limité, robustesse inter-annuelle
- **Horizon** : Épisode complet = 1 année de données M15

## 2. Données
- **Source** : Bougies M15 agrégées (T02) + features techniques (T05)
- **Qualité** : Nettoyées, régulières, sans prix négatifs
- **Alignement** : Split temporel strict (2022 train, 2023 val, 2024 test)
- **Coûts** : Transaction cost = 0.0002 (2 pips) appliqué à chaque trade

## 3. State (Observation)
Fenêtre glissante de N=20 bougies, chaque bougie décrite par :
- **Prix normalisés** : `return_1`, `return_4` (rendements courts)
- **Tendance** : `ema_20`, `ema_50`, `ema_diff` (normalisés par le prix)
- **Momentum** : `rsi_14` (normalisé [0,1])
- **Volatilité** : `rolling_std_20`, `range_15m`
- **Structure bougie** : `body`, `upper_wick`, `lower_wick`
- **Position courante** : -1, 0, +1 (encodée dans le state)

**Normalisation** : Z-score sur fenêtre glissante pour éviter le data leakage.
**Warm-up** : 50 premières bougies ignorées pour stabiliser les indicateurs.

## 4. Action
- **Espace** : `Discrete(3)` → {0: HOLD, 1: BUY, 2: SELL}
- **Exécution** : Le signal change la position (flat/long/short)

## 5. Reward
Reward à chaque pas de temps :
```
reward = position * (price_t - price_{t-1}) / price_{t-1}  # PnL réalisé
         - transaction_cost * |change_of_position|           # coût si trade
         - drawdown_penalty                                  # pénalité MDD
```
- PnL > 0 encourage les bonnes décisions directionnelles
- Pénalité de coût réduit l'overtrading
- Pénalité drawdown encourage la gestion du risque

## 6. Environnement
- **Type** : Simulateur custom compatible `gymnasium.Env`
- **Slippage** : Non modélisé (données M15 suffisamment liquides)
- **Transaction cost** : 0.0002 (2 pips) par trade
- **Épisode** : 1 passage complet sur les données d'une année

## 7. Choix d'algorithme : PPO
**Justification** :
- **Stabilité** : PPO est robuste aux hyperparamètres, idéal pour un premier modèle
- **Exploration** : La politique stochastique explore naturellement l'espace des actions
- **Scalabilité** : Fonctionne bien sur des espaces d'actions discrets
- **Référence** : Utilisé avec succès dans la littérature RL financière

**Alternative testée** : DQN (si PPO ne converge pas)
