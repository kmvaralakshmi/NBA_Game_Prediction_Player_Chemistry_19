import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

# --- Constants and Setup ---
# Define the paths for the generated artifacts
PROCESSED_DATA_PATH = "data/processed_data.npz"
MODEL_WEIGHTS_PATH = "outputs/model_weights.npz"

# --- Utility Functions for Data Loading ---

@st.cache_data
def load_data():
    """
    Loads player list and game logs for display.
    """
    all_players = []
    df_game_log = pd.DataFrame()

    # 1. Load the list of all players (crucial for feature names)
    try:
        with np.load(PROCESSED_DATA_PATH, allow_pickle=True) as data:
            # We assume 'all_players' is the last item saved in the preprocess step
            # which is an array of player names (strings).
            all_players = data['all_players'].tolist()
            # The 'Chemistry' score is the last feature added in preprocess.py
            all_feature_names = all_players + ['Chemistry_Score_Feature']

    except FileNotFoundError:
        st.error(f"❌ Data file not found: {PROCESSED_DATA_PATH}. Please run `python preprocess.py` first.")
        return all_players, df_game_log, None, None, None

    # 2. Load trained model weights (W, S, A)
    try:
        weights = np.load(MODEL_WEIGHTS_PATH)
        W = weights['w']
        S = weights['S']
        A = weights['A']
    except FileNotFoundError:
        st.warning(f"⚠️ Model weights not found at {MODEL_WEIGHTS_PATH}. Displaying mock data for S, A, W. Please run `python chemistry_model.py` to train the real model.")
        # Fallback to mock data if not trained
        n_features = len(all_feature_names)
        W = np.random.randn(n_features) * 0.1
        S = (np.random.rand(n_features, n_features) * 0.1)
        S = (S + S.T) / 2 # Ensure symmetry for mock
        A = (np.random.rand(n_features, n_features) * 0.1)
        A = (A - A.T) / 2 # Ensure anti-symmetry for mock
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return all_players, df_game_log, None, None, None


    # 3. Load the raw game log data for the data table display (assuming a standard path)
    try:
        df_game_log = pd.read_csv("data/nba_games_24_25.csv")
        df_game_log = df_game_log.drop(columns=['home_win'], errors='ignore') # Drop internal cols
    except FileNotFoundError:
        st.warning("Game log data (nba_games_24_25.csv) not found. Game log table will be empty.")


    return all_players, df_game_log, W, S, A, all_feature_names


# --- Load all necessary data ---
all_players, df_game_log, W, S, A, all_feature_names = load_data()
n_players = len(all_players)


# Ensure the app stops if crucial files are missing and couldn't be mocked
if not all_players:
    st.stop()


# --- Streamlit Dashboard Layout ---
st.set_page_config(layout="wide", page_title="NBA Player Chemistry Predictor")

st.title("🏀 NBA Player Chemistry and Synergy Analysis")
st.markdown("---")

# -----------------------------
# 1. Model Overview and Game Log
# -----------------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Overview")
    st.metric(label="Model Used", value="Quadratic Chemistry Model (QCM)")
    st.metric(label="Features", value=f"{len(all_feature_names)} (Players + Chemistry Score)")
    st.markdown("""
    The QCM predicts game outcomes based on a linear term (individual impact) and a quadratic term (player-pair chemistry).
    * **$W$**: Individual Player Impact (Linear Term)
    * **$S$**: Synergy Matrix (Symmetric, $\mathbf{x}^T \mathbf{S} \mathbf{x}$)
    * **$A$**: Rivalry/Anti-Synergy Matrix (Anti-Symmetric, $\mathbf{x}^T \mathbf{A} \mathbf{x}$)
    """)

with col2:
    st.subheader("Latest Game Data")
    st.caption(f"Showing the first 10 rows of the raw game log used for training.")
    if not df_game_log.empty:
        st.dataframe(df_game_log.head(10), use_container_width=True)
    else:
        st.info("Game log data is missing.")

st.markdown("---")


# -----------------------------
# 2. Synergy Matrix (S) - Positive Chemistry
# -----------------------------
st.subheader("🤝 Player Synergy Matrix ($\mathbf{S}$)")
st.caption("The symmetric term of the quadratic model, highlighting positive (synergy) and negative (anti-synergy) fixed-pair impacts on winning. Players are ranked by their average chemistry score across all pairs.")

# To summarize S, we calculate the average synergy of each player with every other player
# The last feature is the 'Chemistry_Score_Feature', so we ignore that for player-player chemistry
S_players = S[:n_players, :n_players]
player_synergy_scores = np.sum(S_players, axis=1)

df_synergy = pd.DataFrame({
    'Player': all_players,
    'Total Synergy Score': player_synergy_scores
}).sort_values(by='Total Synergy Score', ascending=False).reset_index(drop=True)

st.markdown("#### Top 10 Most Synergistic Players (By Total Score)")
st.dataframe(df_synergy.head(10), use_container_width=True)


# --- Detailed Synergy Matrix Visualization (Heatmap) ---
st.markdown("#### Top 10 High Synergy Pairs (S Matrix Values)")

# Create a flat list of all player pairs and their S-scores
synergy_pairs = []
for i in range(n_players):
    for j in range(i + 1, n_players): # Only upper triangle for unique pairs
        synergy_pairs.append({
            'Player 1': all_players[i],
            'Player 2': all_players[j],
            'Synergy Score (S)': S_players[i, j]
        })

df_pairs = pd.DataFrame(synergy_pairs).sort_values(by='Synergy Score (S)', ascending=False)

# Top 10 Synergistic Pairs
df_top_synergy = df_pairs.head(10).style.background_gradient(cmap='Greens')
st.markdown("##### Top Synergy (Positive S-Score)")
st.dataframe(df_top_synergy, use_container_width=True)

# Top 10 Rivalry Pairs (Anti-Synergy)
df_top_rivalry = df_pairs.tail(10).sort_values(by='Synergy Score (S)', ascending=True).style.background_gradient(cmap='Reds')
st.markdown("##### Top Rivalry (Negative S-Score)")
st.dataframe(df_top_rivalry, use_container_width=True)


st.markdown("---")


# -----------------------------
# 3. Individual Player Impact (W Vector)
# -----------------------------
st.subheader("⭐ Individual Player Impact ($\mathbf{W}$ Vector)")
st.caption("Visualization of the linear term ($\mathbf{W}$ vector) from the QCM. This is the player's predicted impact on winning, independent of who they play with.")

# W vector contains the player weights + 1 weight for the Chemistry Score feature (the last feature)
# Filter out the non-player feature for the main chart
df_w = pd.DataFrame({
    'Player': all_feature_names,
    'Impact Score': W
})

df_w_players = df_w[df_w['Player'] != 'Chemistry_Score_Feature'].sort_values(by='Impact Score', ascending=False)

# Select Top 10 Positive and Negative for display
df_w_top = pd.concat([df_w_players.head(10), df_w_players.tail(10)])

# Use the Impact Score of the Chemistry feature as a key metric
chemistry_impact = df_w[df_w['Player'] == 'Chemistry_Score_Feature']['Impact Score'].iloc[0]

col_w1, col_w2 = st.columns([2, 1])

with col_w1:
    fig_w = px.bar(
        df_w_top,
        x='Impact Score',
        y='Player',
        orientation='h',
        color='Impact Score',
        color_continuous_scale=px.colors.diverging.RdBu,
        title='Top/Bottom 10 Players by Individual Impact'
    )
    fig_w.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_w, use_container_width=True)

with col_w2:
    st.metric(
        label="Overall Chemistry Score Feature Impact (W)",
        value=f"{chemistry_impact:.4f}",
        delta="Positive means more players on the court generally increases win probability." if chemistry_impact > 0 else "Negative means less players on the court generally decreases win probability."
    )
    st.markdown("The Chemistry Score is a simple count of players used in `preprocess.py` and its weight in $W$ captures its overall predictive value.")

st.markdown("---")

st.subheader("🔄 Player Rivalry/Anti-Synergy Matrix ($\mathbf{A}$)")
st.caption("The anti-symmetric term of the quadratic model. The magnitude of this term indicates a player's impact is dependent on the context of who they play with (e.g., player A's benefit from playing with player B is not the same as player B's benefit from playing with player A). This term is often harder to interpret directly.")

# Calculate the average magnitude of the anti-symmetric effect for each player
A_players = A[:n_players, :n_players]
player_rivalry_scores = np.sum(np.abs(A_players), axis=1) # Sum of absolute values

df_rivalry = pd.DataFrame({
    'Player': all_players,
    'Total Anti-Symmetry Magnitude': player_rivalry_scores
}).sort_values(by='Total Anti-Symmetry Magnitude', ascending=False).reset_index(drop=True)

st.markdown("#### Top 10 Most Context-Dependent Players (By Total Anti-Symmetry Magnitude)")
st.dataframe(df_rivalry.head(10), use_container_width=True)

st.info("To use this dashboard fully, ensure you have successfully run `python preprocess.py` and `python chemistry_model.py` to generate `data/processed_data.npz` and `outputs/model_weights.npz` with the trained data.")
