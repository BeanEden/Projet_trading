       PROJET FIL ROUGE – VERSION 2
        Système de décision de trading GBP/USD
      (M1 → M15 → ML → RL → API → Docker)


                                      Février 2026


Table des matières

1 Contexte général                                                                        2

2 Données                                                                                 2
  2.1 Période disponible . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    2
  2.2 Split temporel obligatoire . . . . . . . . . . . . . . . . . . . . . . . . . . .    2

3 Structure imposée du projet                                                             2
  3.1 Phase 1 – Importation M1 . . . . . . . . . . . . . . . . . . . . . . . . . . .      2
  3.2 Phase 2 – Agrégation M1 → M15 . . . . . . . . . . . . . . . . . . . . . . .         3
  3.3 Phase 3 – Nettoyage M15 . . . . . . . . . . . . . . . . . . . . . . . . . . .       3

4 Analyse exploratoire                                                                    3

5 Feature Engineering – Version 2                                                         3
  5.1 Bloc court terme . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    3
  5.2 Bloc Contexte & Régime . . . . . . . . . . . . . . . . . . . . . . . . . . . .      4

6 Baseline obligatoire                                                                    4

7 Machine Learning                                                                        4

8 Reinforcement Learning                                                                  5
  8.1 Conception obligatoire sur papier . . . . . . . . . . . . . . . . . . . . . . .     5
  8.2 Paramètres clés . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   5

9 Évaluation finale                                                                       5

10 Industrialisation                                                                      6




                                             1
1      Contexte général
   Vous concevez un système de décision algorithmique sur la paire GBP/USD.
   Fréquence brute : 1 minute
Fréquence décision : 15 minutes
   À chaque décision :
    — BUY
    — SELL
    — HOLD
     Objectif : maximiser le profit cumulé sous contraintes réalistes :
    — coûts de transaction
    — drawdown limité
    — robustesse inter-annuelle
    — décisions mesurées


2      Données
2.1     Période disponible
    — 2022
    — 2023
    — 2024

2.2     Split temporel obligatoire
     Interdiction de split aléatoire.
    — 2022 : Entraînement
    — 2023 : Validation
    — 2024 : Test final (jamais utilisé pour entraîner)
     Walk-forward autorisé si documenté.


3      Structure imposée du projet
3.1     Phase 1 – Importation M1
    — Fusion date + time → timestamp
    — Vérification régularité 1 minute
    — Tri chronologique
    — Détection incohérences




                                              2
3.2     Phase 2 – Agrégation M1 → M15
     Aucune modélisation autorisée en M1.

                            Variable   Règle
                            open_15m open 1ère minute
                            high_15m max(high) sur 15 minutes
                            low_15m   min(low) sur 15 minutes
                            close_15m close dernière minute


3.3     Phase 3 – Nettoyage M15
    — Suppression bougies incomplètes
    — Contrôle prix négatifs
    — Détection gaps anormaux


4      Organisation du travail (mode sprint léger) et Git
4.1     Principe
     Vous travaillez en mode sprint léger (sans Scrum formel) :
    — vous découpez le projet en tâches claires,
    — vous répartissez les tâches (1 ou 2 étudiants par groupe),
    — vous poussez sur Git à chaque tâche terminée (pas de « gros push final »).

4.2     Règles Git obligatoires
    — Un dépôt Git par groupe, avec historique lisible.
    — Une branche par tâche (feature branch).
    — Chaque tâche doit apparaître sur Git via commits réguliers.
    — Chaque étudiant doit pousser au moins une branche (même en binôme).

4.3     Convention de nommage des branches
     Objectif : que l’enseignant sache qui a poussé quoi et pour quelle tâche.
     Format obligatoire :

                             <prenomnom>__<Txx>__<mot-cle>

où :
    — <prenomnom> = identifiant court (ex : aya, marc, ines)
    — <Txx> = numéro de tâche (ex : T01, T06)
    — <mot-cle> = résumé court (ex : m15_agg, features_pack, api_predict)
     Exemples :
    — aya__T01__import_m1

                                               3
    — ines__T02__m15_agg
    — marc__T05__features_regime
    — aya__T08__rl_env
    — ines__T10__api_predict

4.4       Convention de commits
     Chaque commit doit décrire une action concrète.
     Format recommandé :

                           [Txx] verbe: description courte

     Exemples :
    — [T02] add: aggregation M1->M15
    — [T05] fix: remove incomplete candles
    — [T10] add: /predict endpoint with model_version

4.5       Table des tâches (backlog minimal)
     Chaque groupe doit remplir cette table avant de coder puis la mettre à jour.

    ID       Tâche                               Responsable       Branche Git
    T01      Import M1 + contrôle régularité
    T02      Agrégation M1→M15
    T03      Nettoyage M15 + rapport qualité
    T04      Analyse exploratoire + ADF/ACF
    T05      Feature Pack V2 (court terme +
             régime)
    T06      Baseline règles + backtest simple
    T07      ML (split temporel + modèles +
             éval)
    T08      RL (env + reward + entraînement)
    T09      Évaluation robuste (benchmarks +
             2024)
    T10      API (contrat + endpoints + char-
             gement modèle)
    T11      Versioning modèle (v1/v2 + regis-
             try)
    T12      Docker + exécution reproductible


5      Analyse exploratoire
     Obligatoire :
    — Distribution des rendements
    — Volatilité dans le temps

                                           4
    — Analyse horaire
    — Autocorrélation
    — Test ADF


6      Feature Engineering – Version 2
     Toutes les features sont calculées uniquement à partir du passé.

6.1     Bloc court terme
    — return_1
    — return_4
    — ema_20
    — ema_50
    — ema_diff
    — rsi_14
    — rolling_std_20
    — range_15m
    — body
    — upper_wick
    — lower_wick

6.2     Bloc Contexte & Régime
Tendance long terme
    — ema_200
    — distance_to_ema200
    — slope_ema50

Régime de volatilité
    — atr_14
    — rolling_std_100
    — volatility_ratio

Force directionnelle
    — adx_14
    — macd
    — macd_signal




                                             5
7      Baseline obligatoire
     Avant ML ou RL :
    — Stratégie règles fixes
    — Stratégie aléatoire
    — Buy & Hold


8      Machine Learning
     Objectif : prédire le mouvement de la prochaine bougie.
                                   (
                                     1 si closet+1 > closet
                                y=
                                     0 sinon
     Exigences :
    — Split temporel strict
    — Modèle baseline
    — Comparaison modèles
    — Métriques statistiques et financières


9      Reinforcement Learning
9.1     Conception obligatoire sur papier
     Avant codage :
    1. Problème métier (objectif, contraintes, horizon)
    2. Données (qualité, alignement, coûts)
    3. State (features, normalisation, warm-up)
    4. Action (discret ou allocation)
    5. Reward (PnL ou PnL ajusté risque)
    6. Environnement (simulateur, slippage, transaction cost)
    7. Choix algorithme (justification obligatoire)

9.2     Paramètres clés
Paramètres de définition
     state, action, reward, horizon, coûts




                                              6
Paramètres d’entraînement
 — γ
 — learning rate
 — exploration ϵ
 — batch size
 — epochs
 — seed

Paramètres d’évaluation
 — split temporel
 — walk-forward
 — Sharpe
 — drawdown
 — stress tests


10     Évaluation finale
  Comparaison obligatoire :
 — Random
 — Règles
 — ML
 — RL
  Métriques :
 — Profit cumulé
 — Maximum drawdown
 — Sharpe simplifié
 — Profit factor
  Un modèle est valide uniquement s’il est robuste sur 2024.


11     Industrialisation
  Architecture minimale :

 project/
 |
 +-- data/
 +-- features/
 +-- models/
 |   +-- v1/
 |   +-- v2/

                                           7
 +--   training/
 +--   evaluation/
 +--   api/
 +--   docker/


  Règles :
 — L’API expose uniquement le meilleur modèle.
 — L’utilisateur ne peut pas relancer l’entraînement.
 — Versioning modèle obligatoire.
 — L’API charge automatiquement la version validée.


Message clé
  Un modèle performant n’est pas celui qui gagne le plus sur 2022.
  C’est celui qui :
 — survit au changement de régime
 — tient compte des coûts
 — évite l’overfitting temporel
 — est reproductible
 — est industrialisable




                                           8


---
# 📘 Documentation Technique et Fonctionnelle du Projet

## 1. Vue d'ensemble du Projet
Ce projet implémente un système complet de décision de trading algorithmique pour la paire GBP/USD. Il couvre l'intégralité du pipeline de données, de l'ingestion brute à la prise de décision automatisée, en passant par le Machine Learning et une interface utilisateur web moderne.

L'objectif est de fournir une plateforme robuste et simple d'accès permettant de :
- **Analyser** des données financières haute fréquence (M1 transformé en M15).
- **Entraîner et Comparer** des modèles d'IA prédictive (Random Forest, Gradient Boosting, etc.).
- **Visualiser** les performances financières et les métriques techniques en toute transparence.
- **Opérer** des prédictions via une API standardisée et une interface ergonomique.

## 2. Architecture Technique
Le système repose sur une architecture micro-services modulaire et robuste :

*   **Frontend (Interface Utilisateur)** : Développé en **Flask** (Python). Il offre une interface "Cocooning Beige" soignée et intuitive pour visualiser les données, lancer des entraînements sans code et consulter les prédictions.
*   **Backend (API)** : Développé en **FastAPI**. Il gère la logique métier "lourde" : chargement dynamique des modèles, inférence rapide, et communication sécurisée avec les données.
*   **Data Science Core** : Centralisé dans un **Master Notebook** unifié (`Master_Trading_Notebook.ipynb`) qui permet de reproduire pas à pas l'importation, le nettoyage, le feature engineering et la modélisation à des fins de recherche.
*   **Conteneurisation** : Architecture prête pour **Docker** pour garantir la portabilité et la reproductibilité quel que soit l'environnement.

## 3. Workflow Data Science (Le "Cœur" du système)
Le traitement des données suit un processus rigoureux et scientifique en 5 étapes, entièrement automatisé :

1.  **Importation & Audit (T01)** : Chargement des données brutes M1 (1 minute) et audits qualité stricts (détection de trous de cotation, doublons, outliers).
2.  **Agrégation (T02)** : Transformation technique des bougies M1 en bougies M15 (15 minutes) pour lisser la volatilité et réduire le bruit de marché.
3.  **Nettoyage (T03)** : Filtrage intelligent des bougies incomplètes (faible volume de ticks) pour garantir la fiabilité statistique des modèles.
4.  **Feature Engineering (T05)** : Création d'indicateurs techniques avancés pour "nourrir" l'IA :
    *   *Dynamique* : RSI (Indice de Force Relative), Rendements logarithmiques.
    *   *Tendance* : Moyennes Mobiles Exponentielles (EMA), MACD, ADX.
    *   *Volatilité* : ATR, Bandes de Bollinger.
5.  **Machine Learning (T07)** : Entraînement de modèles supervisés avec optimisation automatique des hyperparamètres (GridSearch) et validation temporelle stricte (Train: 2022, Val: 2023, Test: 2024) pour éviter le surapprentissage.

## 4. Guide d'Installation et de Démarrage

### Prérequis
*   Python 3.8 ou supérieur
*   Pip (gestionnaire de paquets Python)
*   Navigateur web récent (Chrome, Firefox, Edge)

### Installation des dépendances
Ouvrez un terminal à la racine du projet et exécutez :
```bash
pip install -r requirements.txt
```

### Lancement de l'application
Le système fonctionne en mode client-serveur. Vous devez lancer deux terminaux distincts.

1.  **Lancer le Backend (API)** :
    Dans le premier terminal :
    ```bash
    python -m uvicorn api.main:app --reload --port 8000
    ```
    *Le backend est prêt quand vous voyez "Application startup complete".*

2.  **Lancer le Frontend (App Web)** :
    Dans le second terminal :
    ```bash
    python -m app.app
    ```
    *Le frontend est prêt quand vous voyez "Running on http://127.0.0.1:5000".*

3.  **Accéder à l'interface** :
    Ouvrez votre navigateur et allez à l'adresse : [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 5. Guide d'Utilisation

Une fois l'interface lancée, vous avez accès à trois zones principales :

*   **🏠 Dashboard** :
    *   Vue d'ensemble de l'état du système.
    *   Indicateurs clés de performance et graphiques sommaires.

*   **👨‍💻 Zone Programmeur (Expert)** :
    *   *Entraînement* : C'est ici que vous créez l'intelligence du système.
        1. Sélectionnez un algorithme (ex: Random Forest, Logistic Regression).
        2. Choisissez les indicateurs (features) à utiliser.
        3. Activez ou non l'optimisation (GridSearch).
        4. Lancez ! Le système gère tout le processus complexe (split temporel, évaluation) et vous affiche les résultats.
    *   *Visualisation* : Analysez la qualité des modèles via les courbes ROC, matrices de confusion et l'importance des variables.

*   **👤 Zone Utilisateur (Trader)** :
    *   *Prédiction* : L'outil d'aide à la décision. Cliquez pour obtenir une recommandation (ACHAT / VENTE / ATTENTE) en temps réel, basée sur le meilleur modèle actuellement entraîné et validé par le système.

## 6. Structure des Dossiers clé
Pour vous repérer dans le code :

*   `api/` : Cerveau du système. Contient le code du backend FastAPI (`main.py`) et la logique de trading.
*   `app/` : Visage du système. Contient le code du frontend Flask (`app.py`) et les fichiers HTML/CSS (`templates/`, `static/`).
*   `data/` : Coffre-fort. Stocke les données brutes (M1), agrégées (M15) et les features calculées.
*   `models/` : Mémoire du système. Sauvegarde automatiquement les modèles entraînés (`.pkl`) et leurs rapports de performance.
*   `notebooks/` : Laboratoire de recherche. Contient les notebooks d'exploration et le `Master_Trading_Notebook.ipynb` pour l'analyse approfondie.