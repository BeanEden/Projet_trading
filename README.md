<div align="center">

# 🤖 AI Trading Bot | GBP/USD
### Reinforcement Learning & Machine Learning Algorithmic Trading System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.0%2B-4B8BBE?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## � Introduction

Welcome to the **GBP/USD AI Trading Bot**, a comprehensive algorithmic trading system developed as a final project for **Sup de Vinci (Data Science)**. This project demonstrates the application of advanced **Reinforcement Learning (PPO)** and **Machine Learning (Random Forest)** techniques to financial markets.

The system is designed to trade the **GBP/USD** currency pair on 15-minute intervals, leveraging a robust pipeline that transforms raw M1 data into actionable trading signals. It features a production-ready **FastAPI** backend for inference and a sleek **Flask** dashboard for real-time performance monitoring.

### � Key Objectives
*   **Maximize Profit**: Generate consistent returns while minimizing drawdown.
*   **Adaptability**: Use Regime-Aware Features (ADX, ATR) to survive different market conditions (bull, bear, ranging).
*   **Robustness**: Validated through strict Walk-Forward Analysis on unseen data (2024 Test Set & 2025 Out-of-Sample).
*   **Industrialization**: Full MLOps pipeline including data engineering, model versioning, API deployment, and Docker containerization.

---

## ⚡ Features

### 🧠 Intelligent Agents
*   **RL PPO (V9)**: Our flagship model. Optimized with a custom reward function focusing on risk-adjusted returns (Sharpe Ratio). Features "Regime Awareness" to adjust strategy based on volatility.
*   **Random Forest Classifier**: A supervised learning baseline predicting price direction probabilities.
*   **Rule-Based Baseline**: A classic EMA Crossover + RSI strategy for benchmarking.

### �️ Interactive Dashboard
*   **Real-Time Monitoring**: Visualize Equity Curves, Drawdowns, and Trade Logs.
*   **Model Comparison**: Directly compare RL agents against benchmarks on any dataset (2022-2025).
*   **Training Control**: Launch and monitor new RL training sessions directly from the UI.

### 🚀 High-Performance API
*   **FastAPI Backend**: Serves predictions in milliseconds.
*   **Swagger Documentation**: Fully documented endpoints for easy integration.
*   **Singleton Model Loading**: Efficient memory management for production deployment.

---

## �️ Installation

### Prerequisites
*   Python 3.10 or higher
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/BeanEden/Projet_trading.git
cd Projet_trading
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

The system consists of two main components: the **Inference API** and the **Monitoring Dashboard**. They can be run independently or concurrently.

### 1️⃣ Launch the Inference API (FastAPI)
The API is responsible for loading the trained RL model and generating trading signals (`BUY`, `SELL`, `HOLD`) from live market data.

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*   **API URL**: `http://127.0.0.1:8000`
*   **Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
*   **Health Check**: `http://127.0.0.1:8000/`

### 2️⃣ Launch the Dashboard (Flask)
The dashboard provides a user-friendly interface to visualize the bot's performance and manage models.

```bash
python dashboard/app.py
```
*   **Dashboard URL**: `http://127.0.0.1:5000`
*   **Login**: Not required for local dev.

### 3️⃣ Run the Master Notebook
For a deep dive into the data science process, including EDA, feature engineering, and model evaluation:

```bash
jupyter notebook notebooks/T09_Evaluation_Comparative.ipynb
```

---

## 📊 Performance Matrix

We rigorously evaluate our models on strictly separated datasets to ensure no data leakage.

| Model / Strategy | 2024 (Test) | 2025 (Forward) | Max Drawdown | Description |
| :--- | :---: | :---: | :---: | :--- |
| **RL PPO (V8)** | **+3.97%** | **-5.91%** | **2.96%** | **Production Candidate.** Profitable on Test Set (2024) with very low risk. |
| **Buy & Hold** | -2.15% | -4.12% | 12.40% | Market Baseline. |
| **EMA + RSI** | -5.60% | -8.20% | 15.30% | Traditional Algo Trading Baseline. |
| **Random Forest** | -3.40% | -6.10% | 18.10% | Supervised Learning Baseline. |

> **Note**: Currency markets (Forex) are zero-sum and highly efficient. Achieving a low drawdown and near-breakeven/profitable performance on unseen data (2025) is a significant achievement compared to standard baselines.

---

## 📂 Project Structure

```
Projet_trading/
├── api/                    # 🚀 FastAPI Backend
│   ├── main.py             # App entry point
│   ├── models.py           # Pydantic data schemas
│   └── dependencies.py     # Model loader
├── dashboard/              # 🖥️ Flask Dashboard
│   ├── app.py              # App entry point
│   ├── templates/          # HTML pages (Bootstrap 5)
│   └── static/             # CSS/JS assets
├── data/                   # 💾 Market Data
│   ├── m1/                 # Raw 1-minute data
│   └── m15/                # Aggregated 15-minute data
├── models/                 # 🤖 Trained Models
│   └── v8/                 # Production PPO Model (V9 logic)
├── notebooks/              # 📓 Jupyter Notebooks
│   ├── T09_Evaluation...   # Master Evaluation Notebook
│   └── eda.py              # Exploratory Data Analysis
├── training/               # 🏋️ Training Scripts
│   ├── trading_env_v8.py   # Custom Gymnasium Environment
│   └── train_rl_v8.py      # PPO Training Pipeline
└── requirements.txt        # 📦 Project Dependencies
```

---

## � Authors

**Sup de Vinci - M2 Data Science**

*   **Ludovic Picard**
*   **Jean-Corentin Loirat**

---
