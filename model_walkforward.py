import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Load all B1 and D1 files
base = '/data/.openclaw/workspace/data/football-data'
def load_league(prefix):
    import glob
    files = sorted(glob.glob(f'{base}/{prefix}*.csv'))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    # Sort by date
    df_all = df_all.sort_values('Date').reset_index(drop=True)
    # Add season column
    df_all['Season'] = df_all['Date'].dt.year.astype(str)
    return df_all

print("Loading B1...")
b1 = load_league('B1')
print(f"B1 matches: {len(b1)}")
print("Loading D1...")
d1 = load_league('D1')
print(f"D1 matches: {len(d1)}")

# Target
b1['over_2.5'] = (b1['FTHG'] + b1['FTAG']) > 2.5
d1['over_2.5'] = (d1['FTHG'] + d1['FTAG']) > 2.5

# Feature engineering function
def compute_features(df):
    df = df.copy()
    # Ensure sorted by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize feature columns
    df['home_goals_scored_home_avg'] = np.nan
    df['home_goals_conceded_home_avg'] = np.nan
    df['away_goals_scored_away_avg'] = np.nan
    df['away_goals_conceded_away_avg'] = np.nan
    df['home_points_form'] = np.nan
    df['away_points_form'] = np.nan
    
    # For each team, compute expanding averages per season
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # We'll use a dictionary to store team-season stats
    # Simpler: compute per match by iterating (slow but ok for this size)
    for idx, row in df.iterrows():
        season = row['Season']
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Previous matches for home team (home games only) in same season
        home_home_prev = df[(df['HomeTeam'] == home) & (df['Season'] == season) & (df.index < idx)]
        if len(home_home_prev) > 0:
            df.loc[idx, 'home_goals_scored_home_avg'] = home_home_prev['FTHG'].mean()
            df.loc[idx, 'home_goals_conceded_home_avg'] = home_home_prev['FTAG'].mean()
        
        # Previous matches for away team (away games only) in same season
        away_away_prev = df[(df['AwayTeam'] == away) & (df['Season'] == season) & (df.index < idx)]
        if len(away_away_prev) > 0:
            df.loc[idx, 'away_goals_scored_away_avg'] = away_away_prev['FTAG'].mean()
            df.loc[idx, 'away_goals_conceded_away_avg'] = away_away_prev['FTHG'].mean()
        
        # Form: points from last 5 matches (any venue) in same season
        home_all_prev = df[((df['HomeTeam'] == home) | (df['AwayTeam'] == home)) & (df['Season'] == season) & (df.index < idx)]
        if len(home_all_prev) > 0:
            home_all_prev = home_all_prev.tail(5)
            points = []
            for _, r in home_all_prev.iterrows():
                if r['HomeTeam'] == home:
                    if r['FTR'] == 'H':
                        points.append(3)
                    elif r['FTR'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
                else:
                    if r['FTR'] == 'A':
                        points.append(3)
                    elif r['FTR'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
            df.loc[idx, 'home_points_form'] = np.mean(points) if points else 0
        
        away_all_prev = df[((df['HomeTeam'] == away) | (df['AwayTeam'] == away)) & (df['Season'] == season) & (df.index < idx)]
        if len(away_all_prev) > 0:
            away_all_prev = away_all_prev.tail(5)
            points = []
            for _, r in away_all_prev.iterrows():
                if r['HomeTeam'] == away:
                    if r['FTR'] == 'H':
                        points.append(3)
                    elif r['FTR'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
                else:
                    if r['FTR'] == 'A':
                        points.append(3)
                    elif r['FTR'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
            df.loc[idx, 'away_points_form'] = np.mean(points) if points else 0
    
    # Fill missing with league averages per season
    for season in df['Season'].unique():
        mask = df['Season'] == season
        for col in ['home_goals_scored_home_avg', 'home_goals_conceded_home_avg',
                   'away_goals_scored_away_avg', 'away_goals_conceded_away_avg']:
            df.loc[mask, col] = df.loc[mask, col].fillna(df.loc[mask, col].mean())
    # Fill remaining with global mean
    for col in ['home_goals_scored_home_avg', 'home_goals_conceded_home_avg',
               'away_goals_scored_away_avg', 'away_goals_conceded_away_avg',
               'home_points_form', 'away_points_form']:
        df[col] = df[col].fillna(df[col].mean())
    
    return df

print("Computing features for B1...")
b1_feat = compute_features(b1)
print("Computing features for D1...")
d1_feat = compute_features(d1)

# Add odds features
for df in [b1_feat, d1_feat]:
    if 'B365>2.5' in df.columns:
        df['odds_over'] = df['B365>2.5']
        df['odds_under'] = df['B365<2.5']
        df['implied_over'] = 1 / df['odds_over']
        df['implied_under'] = 1 / df['odds_under']
        df['overround'] = df['implied_over'] + df['implied_under']
        df['norm_implied_over'] = df['implied_over'] / df['overround']
        df['norm_implied_under'] = df['implied_under'] / df['overround']
    else:
        df['odds_over'] = np.nan
        df['norm_implied_over'] = np.nan

# Feature columns
feature_cols = [
    'home_goals_scored_home_avg',
    'home_goals_conceded_home_avg',
    'away_goals_scored_away_avg',
    'away_goals_conceded_away_avg',
    'home_points_form',
    'away_points_form',
    'norm_implied_over'  # include market information
]

# Drop rows missing any feature or target
train = b1_feat.dropna(subset=feature_cols + ['over_2.5'])
test = d1_feat.dropna(subset=feature_cols + ['over_2.5'])

print(f"Training samples: {len(train)}")
print(f"Test samples: {len(test)}")

X_train = train[feature_cols].values
y_train = train['over_2.5'].values
X_test = test[feature_cols].values
y_test = test['over_2.5'].values

# Model pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42, max_iter=1000)
)
model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]
test['pred_prob_over'] = y_pred_proba

# Evaluate classification performance
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
acc = accuracy_score(y_test, y_pred_proba > 0.5)
auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)
print(f"\nModel performance on D1 (unseen league):")
print(f"  Accuracy: {acc:.3f}")
print(f"  ROC‑AUC:  {auc:.3f}")
print(f"  Brier:    {brier:.3f}")

# Betting simulation
def simulate_betting(df, prob_col='pred_prob_over', odds_col='odds_over', threshold=0.0, bankroll=1000, bet_size=0.02):
    df = df.copy()
    df['bet'] = df[prob_col] > df['norm_implied_over'] + threshold
    df['stake'] = bankroll * bet_size * df['bet']
    df['profit'] = 0.0
    for idx, row in df.iterrows():
        if row['bet']:
            if row['over_2.5']:
                df.loc[idx, 'profit'] = row['stake'] * (row[odds_col] - 1)
            else:
                df.loc[idx, 'profit'] = -row['stake']
    df['cumulative'] = df['profit'].cumsum() + bankroll
    total_bets = df['bet'].sum()
    total_profit = df['profit'].sum()
    roi = total_profit / df['stake'].sum() if df['stake'].sum() > 0 else 0
    return df, total_bets, total_profit, roi

print("\n--- Betting Simulation vs Bet365 ---")
# Threshold 0 means bet whenever model probability > implied probability
test_sim, n_bets, profit, roi = simulate_betting(test, threshold=0.0, bankroll=1000, bet_size=0.02)
print(f"Bets placed: {n_bets} out of {len(test)} matches")
print(f"Total profit: {profit:.2f} units")
print(f"ROI: {roi:.2%}")
print(f"Final bankroll: {test_sim['cumulative'].iloc[-1]:.2f}")

# Compare to naive strategies
print("\n--- Comparison to Naive Strategies ---")
# 1. Always bet over 2.5
always_over = test.copy()
always_over['profit'] = np.where(always_over['over_2.5'], (always_over['odds_over'] - 1) * 20, -20)
always_over_profit = always_over['profit'].sum()
print(f"Always bet over 2.5 (20 units per match): profit {always_over_profit:.2f}")

# 2. Bet on over 2.5 when implied probability < 0.5 (odds > 2.0)
value_bet = test.copy()
value_bet['bet'] = value_bet['norm_implied_over'] < 0.5
value_bet['profit'] = 0
for idx, row in value_bet.iterrows():
    if row['bet']:
        if row['over_2.5']:
            value_bet.loc[idx, 'profit'] = (row['odds_over'] - 1) * 20
        else:
            value_bet.loc[idx, 'profit'] = -20
value_bet_profit = value_bet['profit'].sum()
print(f"Bet over when odds > 2.0 (implied < 0.5): profit {value_bet_profit:.2f}")

# 3. Random betting
np.random.seed(42)
random_bets = np.random.choice([True, False], size=len(test))
random_profit = 0
for idx, r in test.iterrows():
    if random_bets[idx]:
        if r['over_2.5']:
            random_profit += (r['odds_over'] - 1) * 20
        else:
            random_profit -= 20
print(f"Random betting (50% chance): profit {random_profit:.2f}")

# Plot cumulative profit
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test_sim['cumulative'].values, label='Model (threshold=0)', color='blue')
ax.axhline(y=1000, color='gray', linestyle='--', label='Initial bankroll')
ax.set_xlabel('Match number')
ax.set_ylabel('Bankroll (units)')
ax.set_title('Walk‑Forward Testing on D1: Cumulative Profit vs Bet365')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plot_path = '/data/.openclaw/workspace/data/football-data/model_profit.png'
plt.savefig(plot_path, dpi=150)
print(f"\nSaved profit chart to {plot_path}")

# Feature importance
if hasattr(model, 'named_steps'):
    coef = model.named_steps['logisticregression'].coef_[0]
else:
    coef = model.coef_[0]
print("\nFeature importance (coefficients):")
for col, c in zip(feature_cols, coef):
    print(f"  {col}: {c:+.4f}")

# Summary
print("\n=== Summary ===")
print(f"Model trained on B1 ({len(train)} matches), tested on D1 ({len(test)} matches).")
print(f"Model beats naive strategies: {profit > max(always_over_profit, value_bet_profit, random_profit)}")
print(f"ROI of model: {roi:.2%}")