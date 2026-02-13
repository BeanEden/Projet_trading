<div align="center">

# Bot de Trading IA | GBP/USD
### Système de Trading Algorithmique par Apprentissage par Renforcement & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.0%2B-4B8BBE?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 1. Introduction

Bienvenue sur le **Bot de Trading IA GBP/USD**, un système complet de trading algorithmique développé comme projet final pour **Sup de Vinci (Data Science)**. Ce projet démontre l'application des techniques avancées d'**Apprentissage par Renforcement (PPO)** et de **Machine Learning (Random Forest)** aux marchés financiers.

Le système est conçu pour trader la paire de devises **GBP/USD** sur des intervalles de 15 minutes, en s'appuyant sur un pipeline robuste qui transforme les données brutes M1 en signaux de trading exploitables. Il dispose d'un backend **FastAPI** pour l'inférence et d'un tableau de bord **Flask** pour le suivi des performances en temps réel.

### Objectifs Clés
*   **Maximiser le Profit** : Générer des rendements constants tout en minimisant le drawdown.
*   **Adaptabilité** : Utilisation de caractéristiques de régime (ADX, ATR) pour survivre à différentes conditions de marché (haussier, baissier, range).
*   **Robustesse** : Validé par une analyse Walk-Forward stricte sur des données inédites (Test 2024 et Out-of-Sample 2025).
*   **Industrialisation** : Pipeline MLOps complet incluant l'ingénierie des données, le versionnage des modèles, le déploiement API et la conteneurisation Docker.

---

## 2. Fonctionnalités

### 1) Agents Intelligents
*   **RL PPO (V8)** : Notre modèle phare. Optimisé avec une fonction de récompense personnalisée axée sur les rendements ajustés au risque (Ratio de Sharpe). Intègre une "conscience du régime" pour ajuster la stratégie en fonction de la volatilité.
*   **Classifieur Random Forest** : Une base d'apprentissage supervisé prédisant les probabilités de direction des prix.
*   **Stratégie de Base** : Une stratégie classique de croisement EMA + RSI pour le benchmarking.

### 2) Tableau de Bord Interactif
*   **Suivi en Temps Réel** : Visualisation des courbes d'équité, des drawdowns et des journaux de transactions.
*   **Comparaison de Modèles** : Comparaison directe des agents RL par rapport aux références sur n'importe quel jeu de données (2022-2025).
*   **Contrôle de l'Entraînement** : Lancement et suivi de nouvelles sessions d'entraînement RL directement depuis l'interface utilisateur.

### 3) API Haute Performance
*   **Backend FastAPI** : Fournit des prédictions en quelques millisecondes.
*   **Documentation Swagger** : Points de terminaison entièrement documentés pour une intégration facile.
*   **Chargement de Modèle Singleton** : Gestion efficace de la mémoire pour le déploiement en production.

---

## 3. Installation

### Prérequis
*   Python 3.10 ou supérieur
*   Git

### 1) Cloner le Dépôt
```bash
git clone https://github.com/BeanEden/Projet_trading.git
cd Projet_trading
```

### 2) Installer les Dépendances
Il est recommandé d'utiliser un environnement virtuel.
```bash
# Créer l'environnement virtuel
python -m venv .venv

# Activer (Windows)
.\.venv\Scripts\activate

# Activer (Linux/Mac)
source .venv/bin/activate

# Installer les packages
pip install -r requirements.txt
```

---

## 4. Guide d'Utilisation

Le système se compose de deux éléments principaux : l'**API d'Inférence** et le **Tableau de Bord de Suivi**. Ils peuvent être exécutés indépendamment ou simultanément.

### 1) Lancer l'API d'Inférence (FastAPI)
L'API est responsable du chargement du modèle RL entraîné et de la génération de signaux de trading (`BUY`, `SELL`, `HOLD`) à partir des données du marché en direct.

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*   **URL de l'API** : `http://127.0.0.1:8000`
*   **Documentation (Swagger UI)** : `http://127.0.0.1:8000/docs`
*   **Test de Santé** : `http://127.0.0.1:8000/`

### 2) Lancer le Tableau de Bord (Flask)
Le tableau de bord offre une interface conviviale pour visualiser les performances du bot et gérer les modèles.

```bash
python dashboard/app.py
```
*   **URL du Tableau de Bord** : `http://127.0.0.1:5000`

### 3) Exécuter le Master Notebook
Pour une immersion profonde dans le processus de science des données, incluant l'EDA, l'ingénierie des caractéristiques et l'évaluation des modèles :

```bash
jupyter notebook notebooks/Master Notebook.ipynb
```

---

## 5. Déploiement avec Docker

Le projet est entièrement conteneurisé, ce qui permet de lancer l'API et le Tableau de Bord en une seule commande, sans avoir à installer de dépendances localement.

### Prérequis
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et lancé.

### Lancement Rapide (Recommandé)
```bash
# Construire et lancer les conteneurs
docker compose up --build
```

Une fois lancé :
*   **Tableau de Bord** : accessible sur `http://localhost:5000`
*   **API (Documentation)** : accessible sur `http://localhost:8000/docs`

Pour arrêter les services :
```bash
docker compose down
```

---

## 6. Matrice de Performance

Nous évaluons rigoureusement nos modèles sur des jeux de données strictement séparés pour garantir l'absence de fuite de données.

| Modèle / Stratégie | 2024 (Test) | 2025 (Forward) | Drawdown Max | Description |
| :--- | :---: | :---: | :---: | :--- |
| **RL PPO (V8)** | **+3.97%** | **-5.91%** | **2.96%** | **Candidat à la Production.** Profitable sur le Test Set (2024) avec un risque très faible. |
| **Buy & Hold** | -2.15% | -4.12% | 12.40% | Référence du marché. |
| **EMA + RSI** | -5.60% | -8.20% | 15.30% | Référence de trading algorithmique traditionnel. |
| **Random Forest** | -3.40% | -6.10% | 18.10% | Référence d'apprentissage supervisé. |

> **Note** : Les marchés des devises (Forex) sont à somme nulle et très efficients. Obtenir un faible drawdown et une performance proche de l'équilibre ou profitable sur des données inédites (2025) est une réalisation significative par rapport aux références standards.

---

## 6. Structure du Projet

```
Projet_trading/
├── api/                    # Backend FastAPI
│   ├── main.py             # Point d'entrée de l'application
│   ├── models.py           # Schémas de données Pydantic
│   └── dependencies.py     # Chargeur de modèle
├── dashboard/              # Tableau de bord Flask
│   ├── app.py              # Point d'entrée de l'application
│   ├── templates/          # Pages HTML (Bootstrap 5)
│   └── static/             # Actifs CSS/JS
├── data/                   # Stockage des données de marché
│   ├── m1/                 # Données brutes 1 minute
│   └── m15/                # Données agrégées 15 minutes
├── models/                 # Modèles entraînés
│   └── v8/                 # Modèle PPO de production
├── notebooks/              # Notebooks Jupyter
│   ├── T09_Evaluation...   # Notebook d'évaluation Master
│   └── eda.py              # Analyse exploratoire des données
├── training/               # Scripts d'entraînement
│   ├── trading_env_v8.py   # Environnement Gymnasium personnalisé
│   └── train_rl_v8.py      # Pipeline d'entraînement PPO
└── requirements.txt        # Dépendances du projet
```

---

## 7. Auteurs

**Sup de Vinci - M2 Data Science**

*   **Ludovic Picard**
*   **Jean-Corentin Loirat**

---
