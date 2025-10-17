import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

# --- Configuration ---
DATA_FILE = "data/nba_games_24_25.csv"
PROCESSED_DATA_FILE = "data/processed_data.npz"

# --- Load dataset ---
try:
    # Assuming 'data' directory is relative to where the script is run
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"❌ Error: Raw data file '{DATA_FILE}' not found. Please ensure it is in the 'data' directory.")
    exit()

# Encode target: Home win = 1, else 0. Assuming 'Res' column indicates result (W/L)
df['home_win'] = df['Res'].apply(lambda x: 1 if x == 'W' else 0)

# Create player lists per team per game (grouping by date, team, and opponent)
group_cols = ["Data", "Tm", "Opp"]
player_lists = []
targets = []

# Group by game/team combination
for _, group in df.groupby(group_cols):
    # Determine the target for this unique game/team combination
    # The first row's result ('Res') determines the win/loss for the team
    target = 1 if group["Res"].iloc[0] == 'W' else 0

    # Get the unique players for this game/team who played
    players = set(group["Player"].dropna())

    # Only include records with players present
    if players:
        player_lists.append(players)
        targets.append(target)

# Flatten all players to create the feature list
all_players = sorted(set().union(*player_lists))

# Binary encoding of players (One-hot encoding for the lineup)
mlb = MultiLabelBinarizer(classes=all_players)
X_players = mlb.fit_transform(player_lists)

# Chemistry feature: number of players per team per game (this is the extra feature)
chemistry = np.array([len(p) for p in player_lists]).reshape(-1, 1)

# Combine player encoding + chemistry feature
X = np.hstack([X_players, chemistry])
y = np.array(targets)

# --- Save processed data ---
Path("data").mkdir(exist_ok=True) # Ensure data directory exists

# *** This line ensures 'all_players' is saved with the key 'all_players' for app.py ***
np.savez_compressed(
    PROCESSED_DATA_FILE,
    X=X,
    y=y,
    # np.array(..., dtype=object) is crucial for saving a list of strings/objects
    all_players=np.array(all_players, dtype=object)
)

print(f"✅ Data preprocessing complete. Processed {len(X)} records.")
print(f"Data saved to {PROCESSED_DATA_FILE} with keys: X, y, all_players.")
