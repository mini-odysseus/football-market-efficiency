import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

base = '/data/.openclaw/workspace/data/football-data'

# Load B1 only for walk-forward analysis
import glob
files = sorted(glob.glob(f'{base}/B1*.csv'))
dfs = []
for f in files:
    if 'analysis' in f or 'efficiency' in f:
        continue
    df = pd.read_csv(f)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values('Date').reset_index(drop=True)
df['Season'] = df['Date'].dt.year.astype(int)
df['over_2.5'] = (df['FTHG'] + df['FTAG']) > 2.5

print(f"Total B1 matches: {len(df)}")
print(f"Seasons: {sorted(df['Season'].unique())}")

# Feature engineering - expanding window per team per season
def add_features(df):
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    # Team statistics using expanding windows (no lookahead)
    df['home_gs_avg'] = np.nan
    df['home_gc_avg'] = np.nan
    df['away_gs_avg'] = np.nan
    df['away_gc_avg'] = np.nan
    df['home_form'] = np.nan
    df['away_form'] = np.nan
    df['home_shots_avg'] = np.nan
    df['away_shots_avg'] = np.nan
    
    for idx, row in df.iterrows():
        season = row['Season']
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Home team home matches before this one
        home_prev = df[(df['HomeTeam'] == home) & (df['Season'] == season) & (df.index < idx)]
        if len(home_prev) >= 3:
            df.loc[idx, 'home_gs_avg'] = home_prev['FTHG'].mean()
            df.loc[idx, 'home_gc_avg'] = home_prev['FTAG'].mean()
            df.loc[idx, 'home_shots_avg'] = home_prev['HS'].mean()
            # Points form (last 5 matches)
            last5 = home_prev.tail(5)
            pts = []
            for _, r in last5.iterrows():
                if r['FTR'] == 'H':
                    pts.append(3)
                elif r['FTR'] == 'D':
                    pts.append(1)
                else:
                    pts.append(0)
            df.loc[idx, 'home_form'] = np.mean(pts)
        elif len(home_prev) > 0:
            df.loc[idx, 'home_gs_avg'] = home_prev['FTHG'].mean()
            df.loc[idx, 'home_gc_avg'] = home_prev['FTAG'].mean()
            df.loc[idx, 'home_shots_avg'] = home_prev['HS'].mean()
            df.loc[idx, 'home_form'] = 1.0  # Neutral prior
        
        # Away team away matches before this one
        away_prev = df[(df['AwayTeam'] == away) & (df['Season'] == season) & (df.index < idx)]
        if len(away_prev) >= 3:
            df.loc[idx, 'away_gs_avg'] = away_prev['FTAG'].mean()
            df.loc[idx, 'away_gc_avg'] = away_prev['FTHG'].mean()
            df.loc[idx, 'away_shots_avg'] = away_prev['AS'].mean()
            # Points form
            last5 = away_prev.tail(5)
            pts = []
            for _, r in last5.iterrows():
                if r['FTR'] == 'A':
                    pts.append(3)
                elif r['FTR'] == 'D':
                    pts.append(1)
                else:
                    pts.append(0)
            df.loc[idx, 'away_form'] = np.mean(pts)
        elif len(away_prev) > 0:
            df.loc[idx, 'away_gs_avg'] = away_prev['FTAG'].mean()
            df.loc[idx, 'away_gc_avg'] = away_prev['FTHG'].mean()
            df.loc[idx, 'away_shots_avg'] = away_prev['AS'].mean()
            df.loc[idx, 'away_form'] = 1.0
    
    # Fill missing with league averages
    for col in ['home_gs_avg', 'home_gc_avg', 'home_shots_avg', 'home_form',
                'away_gs_avg', 'away_gc_avg', 'away_shots_avg', 'away_form']:
        df[col] = df[col].fillna(df[col].mean())
    
    # Market features
    df['implied_over'] = 1 / df['B365>2.5']
    df['implied_under'] = 1 / df['B365<2.5']
    df['overround'] = df['implied_over'] + df['implied_under']
    df['norm_implied_over'] = df['implied_over'] / df['overround']
    
    # Derived features
    df['goal_diff_home'] = df['home_gs_avg'] - df['away_gc_avg']
    df['goal_diff_away'] = df['away_gs_avg'] - df['home_gc_avg']
    df['total_expected'] = df['home_gs_avg'] + df['away_gs_avg']
    
    return df

print("\nEngineering features...")
df = add_features(df)

# Train/test split by season (walk-forward)
train_seasons = [2020, 2021, 2022]
test_seasons = [2023, 2024, 2025]

train_mask = df['Season'].isin(train_seasons)
test_mask = df['Season'].isin(test_seasons)

feature_cols = [
    'home_gs_avg', 'home_gc_avg', 'home_form', 'home_shots_avg',
    'away_gs_avg', 'away_gc_avg', 'away_form', 'away_shots_avg',
    'norm_implied_over', 'goal_diff_home', 'goal_diff_away', 'total_expected'
]

train_df = df[train_mask].dropna(subset=feature_cols + ['over_2.5'])
test_df = df[test_mask].dropna(subset=feature_cols + ['over_2.5'])

print(f"\nTraining samples (2020-2022): {len(train_df)}")
print(f"Test samples (2023-2025): {len(test_df)}")

X_train = train_df[feature_cols].values
y_train = train_df['over_2.5'].values
X_test = test_df[feature_cols].values
y_test = test_df['over_2.5'].values

# Train model
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
model.fit(X_train, y_train)

# Predict
y_pred_proba = model.predict_proba(X_test)[:, 1]
test_df = test_df.copy()
test_df['pred_prob_over'] = y_pred_proba

# Performance metrics
acc = accuracy_score(y_test, y_pred_proba > 0.5)
auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print(f"\n=== Model Performance (Walk-Forward on 2023-2025) ===")
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC:  {auc:.3f}")
print(f"Brier:    {brier:.3f}")

# Betting simulation - bet when model confidence exceeds market threshold
def simulate_betting(df, threshold=0.05, flat_stake=20):
    """Bet on over 2.5 when model prob > implied prob + threshold"""
    df = df.copy()
    df['bet'] = df['pred_prob_over'] > (df['norm_implied_over'] + threshold)
    df['stake'] = np.where(df['bet'], flat_stake, 0)
    
    # Profit calculation
    df['profit'] = 0.0
    win_mask = df['bet'] & df['over_2.5']
    lose_mask = df['bet'] & ~df['over_2.5']
    df.loc[win_mask, 'profit'] = df.loc[win_mask, 'stake'] * (df.loc[win_mask, 'B365>2.5'] - 1)
    df.loc[lose_mask, 'profit'] = -df.loc[lose_mask, 'stake']
    
    df['cumulative'] = 1000 + df['profit'].cumsum()
    
    total_bets = df['bet'].sum()
    total_profit = df['profit'].sum()
    roi = total_profit / (total_bets * flat_stake) if total_bets > 0 else 0
    
    return df, total_bets, total_profit, roi

print("\n=== Betting Results vs Bet365 ===")

# Test different thresholds
for thresh in [0.0, 0.03, 0.05, 0.08, 0.10]:
    sim_df, n_bets, profit, roi = simulate_betting(test_df, threshold=thresh, flat_stake=20)
    final_bankroll = 1000 + profit
    print(f"Threshold +{thresh:.0%}: {n_bets} bets, profit {profit:+.0f}, ROI {roi:+.1%}, final {final_bankroll:.0f}")

# Best threshold simulation
best_thresh = 0.05
sim_df, n_bets, profit, roi = simulate_betting(test_df, threshold=best_thresh, flat_stake=20)

# Compare to naive strategies
print("\n=== Comparison to Naive Strategies ===")

# Always bet over
always_over = test_df.copy()
always_over['profit'] = np.where(always_over['over_2.5'], 
                                  20 * (always_over['B365>2.5'] - 1), -20)
avg_over_profit = always_over['profit'].sum()
print(f"Always bet over:  {avg_over_profit:+.0f} profit")

# Kelly criterion (fractional)
def kelly_bet(p, odds, fraction=0.25):
    """Fractional Kelly: bet fraction of full Kelly"""
    q = 1 - p
    b = odds - 1
    kelly = (b * p - q) / b if b > 0 else 0
    return max(0, kelly * fraction)

kelly_df = test_df.copy()
kelly_df['kelly_fraction'] = kelly_df.apply(
    lambda r: kelly_bet(r['pred_prob_over'], r['B365>2.5'], 0.25), axis=1)
kelly_df['stake'] = kelly_df['kelly_fraction'] * 1000  # % of bankroll
kelly_df['stake'] = np.clip(kelly_df['stake'], 0, 50)  # Max 50 units
kelly_df['profit'] = 0.0
for idx, row in kelly_df.iterrows():
    if row['stake'] > 0:
        if row['over_2.5']:
            kelly_df.loc[idx, 'profit'] = row['stake'] * (row['B365>2.5'] - 1)
        else:
            kelly_df.loc[idx, 'profit'] = -row['stake']
kelly_profit = kelly_df['profit'].sum()
print(f"Kelly criterion:  {kelly_profit:+.0f} profit")

# Plot cumulative profit for best strategy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sim_df['cumulative'].values, label=f'Model (thresh +{best_thresh:.0%})', linewidth=2)
ax.axhline(y=1000, color='gray', linestyle='--', label='Initial bankroll')
ax.axhline(y=sim_df['cumulative'].iloc[-1], color='blue', linestyle=':', alpha=0.5)
ax.set_xlabel('Match Number (2023-2025 season)')
ax.set_ylabel('Bankroll (units)')
ax.set_title('Walk-Forward Profitability: B1 Model on 2023-2025 Seasons vs Bet365')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plot_path = '/data/.openclaw/workspace/data/football-data/walkforward_profit.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nSaved profit chart to {plot_path}")

# Feature importance
coef = model.named_steps['logisticregression'].coef_[0]
importance = pd.DataFrame({'feature': feature_cols, 'coef': coef})
importance = importance.reimportance = importance.reindex(importance['coef'].abs().sort_values(ascending=False).index)
print("\n=== Top Features ===")
print(importance.head(8).to_string(index=False))

# Per-season breakdown
print("\n=== Season-by-Season Results (threshold +5%) ===")
for season in sorted(test_df['Season'].unique()):
    season_df = test_df[test_df['Season'] == season]
    sim_s, bets, prof, roi = simulate_betting(season_df, threshold=0.05, flat_stake=20)
    acc_s = accuracy_score(season_df['over_2.5'], season_df['pred_prob_over'] > 0.5)
    print(f"Season {season}: {len(season_df)} matches, acc {acc_s:.2f}, {bets} bets, {prof:+.0f} profit, {roi:+.1%} ROI")