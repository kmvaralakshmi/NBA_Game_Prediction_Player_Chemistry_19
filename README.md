NBA Game Predictions Based on Player Chemistry

Overview

This project predicts the outcomes of NBA basketball games (Win/Loss) by explicitly leveraging player chemistry — how well players perform together as a team.
It is inspired by the Stanford CS229 research paper
“Predicting NBA Game Outcomes Using Player Chemistry” (2019).

We implement and compare four distinct models:

Logistic Regression (Baseline): A simple linear model.

Quadratic Player Chemistry Model (Static): A custom model capturing player synergy and rivalry.

Dynamic Chemistry Model: An advanced version of the custom model that weights chemistry over time.

Random Forest Classifier (Benchmark): A strong non-linear model for performance comparison.

Problem Statement

Predict whether the home team wins an NBA game using both teams’ player lineups.
The project explores how player synergy and chemistry impact the outcome beyond individual statistics and serves as a proof-of-concept for the chemistry feature.

Objective

Collect and process NBA game data with team/player details.

Engineer features based on player combinations (Team Chemistry Index - TCI).

Train and evaluate multiple models, including custom chemistry models and benchmarks.

Visualize predictions and chemistry factors using a Streamlit dashboard to interpret the effect of team cohesion.

🧩 Project Structure

NBA_Game_Prediction_TeamID/
├── data/
│ ├── nba_games_24_25.csv # Raw dataset
│ └── processed_data.npz # Preprocessed feature matrix (X, y)
├── outputs/
│ ├── random_forest_model.pkl # Saved Random Forest model
│ ├── y_pred.npy # Model predictions for evaluation
│ └── ... # Other model outputs and SHAP plots
├── src/
│ ├── preprocess.py
│ ├── logistic_baseline.py # Baseline Model
│ ├── chemistry_model.py # Custom Static Chemistry Model
│ ├── dynamic_chemistry_model.py # Custom Dynamic Model
│ ├── random_forest_benchmark.py # NEW: Saved Benchmark Model
│ ├── evaluate.py
│ └── explainability.py # SHAP analysis
├── run_all.ps1 # Automated training pipeline
├── app.py # Streamlit Dashboard (Main UI)
└── README.md

🚀 End-to-End Run Summary

To set up and run the entire project pipeline:

# 1. Install dependencies

pip install -r requirements.txt

# 2. Run the full training pipeline (runs all src/\*.py files sequentially)

./run_all.ps1

# 3. Launch the interactive Streamlit dashboard

python -m streamlit run app.py

📊 Model Performance Comparison

Model

Type

Expected Accuracy

Purpose

Logistic Regression

Linear

60%

Baseline comparison

Quadratic Chemistry Model (Static)

Custom

62%

Proof of Chemistry Concept

Dynamic Chemistry Model

Custom

63%

Measures time decay of chemistry

Random Forest Classifier

Non-Linear

64%

Strong non-linear benchmark

🧠 Key Takeaways

The Team Chemistry Index (TCI) is a measurable factor that influences game outcomes significantly.

Non-linear models (Random Forest) and custom quadratic models often outperform simple linear baselines.

The dashboard allows for visual validation that higher TCI directly correlates with higher win probability.

This framework can be extended to use real-time player statistics and larger datasets.

📚 Reference

Stanford CS229 Project Report (2019):
Predicting NBA Game Outcomes Using Player Chemistry
