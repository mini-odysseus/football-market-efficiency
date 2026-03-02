import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, glob, sys

# Load all D1 files
base = '/data/.openclaw/workspace/data/football-data'
d1_files = sorted(glob.glob(os.path.join(base, 'D1*.csv')))
d1_data = []
for f in d1_files:
    df = pd.read_csv(f)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Season'] = df['Date'].dt.year.astype(str)
    d1_data.append(df)
d1 = pd.concat(d1_data, ignore_index=True)
print(f"D1 total matches: {len(d1)}")

# Hypothesis 1: Home advantage trend over seasons
home_win_rate = d1.groupby('Season')['FTR'].apply(lambda x: (x == 'H').mean())
draw_rate = d1.groupby('Season')['FTR'].apply(lambda x: (x == 'D').mean())
away_win_rate = d1.groupby('Season')['FTR'].apply(lambda x: (x == 'A').mean())

print("Home win rate by season:")
print(home_win_rate.round(3))

# Hypothesis 2: Over 2.5 goals betting odds vs actual over‑2.5 frequency
if 'B365>2.5' in d1.columns:
    d1['over_2.5_actual'] = (d1['FTHG'] + d1['FTAG']) > 2.5
    over_rate = d1.groupby('Season')['over_2.5_actual'].mean()
    avg_odds = d1.groupby('Season')['B365>2.5'].mean()
    print("Over 2.5 actual rate vs avg odds:")
    for season in sorted(over_rate.index):
        print(f"  {season}: {over_rate[season]:.3f} actual, odds {avg_odds[season]:.2f}")
    # Compute implied probability (1/odds)
    d1['implied_prob_over'] = 1 / d1['B365>2.5']
    # Bookmaker margin: sum(1/odds) > 1
    margin = d1.groupby('Season').apply(lambda g: (1/g['B365>2.5'] + 1/g['B365<2.5']).mean())
    print("Avg bookmaker margin (over+under):")
    for season, m in margin.items():
        print(f"  {season}: {m:.3f}")

# Hypothesis 3: Goal difference distribution
d1['GD'] = d1['FTHG'] - d1['FTAG']
gd_counts = d1['GD'].value_counts().sort_index()
print("Goal difference distribution (sample):")
print(gd_counts.head(10))

# Plot 1: Home win rate over seasons
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].plot(home_win_rate.index, home_win_rate.values, marker='o', label='Home win')
axes[0,0].plot(draw_rate.index, draw_rate.values, marker='s', label='Draw')
axes[0,0].plot(away_win_rate.index, away_win_rate.values, marker='^', label='Away win')
axes[0,0].set_title('Result distribution by season (D1 Bundesliga)')
axes[0,0].set_xlabel('Season')
axes[0,0].set_ylabel('Proportion')
axes[0,0].legend()
axes[0,0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Avg goals per match over seasons
goals_per_match = d1.groupby('Season').apply(lambda g: (g['FTHG'].sum() + g['FTAG'].sum()) / len(g))
axes[0,1].plot(goals_per_match.index, goals_per_match.values, marker='o', color='green')
axes[0,1].set_title('Avg total goals per match (D1)')
axes[0,1].set_xlabel('Season')
axes[0,1].set_ylabel('Goals')
axes[0,1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Over 2.5 actual vs implied probability
if 'over_2.5_actual' in d1.columns:
    season_over = d1.groupby('Season')['over_2.5_actual'].mean()
    season_implied = d1.groupby('Season')['implied_prob_over'].mean()
    axes[1,0].plot(season_over.index, season_over.values, marker='o', label='Actual')
    axes[1,0].plot(season_implied.index, season_implied.values, marker='s', label='Implied (1/odds)')
    axes[1,0].set_title('Over 2.5 goals: actual vs implied probability')
    axes[1,0].set_xlabel('Season')
    axes[1,0].set_ylabel('Probability')
    axes[1,0].legend()
    axes[1,0].grid(True, linestyle='--', alpha=0.6)

# Plot 4: Goal difference histogram
axes[1,1].hist(d1['GD'], bins=range(-8, 9), edgecolor='black', alpha=0.7)
axes[1,1].set_title('Goal difference distribution (D1)')
axes[1,1].set_xlabel('Home goals – Away goals')
axes[1,1].set_ylabel('Frequency')
axes[1,1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plot_path = '/data/.openclaw/workspace/data/football-data/d1_analysis.png'
plt.savefig(plot_path, dpi=150)
print(f"Saved plot to {plot_path}")
plt.close()

# Now produce a comparison plot B1 vs D1
# Load B1 again
b1_files = sorted(glob.glob(os.path.join(base, 'B1*.csv')))
b1_data = []
for f in b1_files:
    df = pd.read_csv(f)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Season'] = df['Date'].dt.year.astype(str)
    b1_data.append(df)
b1 = pd.concat(b1_data, ignore_index=True)

# Compute metrics per league per season
def compute_metrics(df):
    home_win = df.groupby('Season')['FTR'].apply(lambda x: (x == 'H').mean())
    over_rate = df.groupby('Season').apply(lambda g: ((g['FTHG'] + g['FTAG']) > 2.5).mean())
    avg_goals = df.groupby('Season').apply(lambda g: (g['FTHG'].sum() + g['FTAG'].sum()) / len(g))
    return pd.DataFrame({'home_win': home_win, 'over_rate': over_rate, 'avg_goals': avg_goals})

b1_metrics = compute_metrics(b1)
d1_metrics = compute_metrics(d1)

# Align seasons
common_seasons = sorted(set(b1_metrics.index) & set(d1_metrics.index))
print("Common seasons:", common_seasons)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
# Home win rate comparison
axes2[0,0].plot(common_seasons, [b1_metrics.loc[s, 'home_win'] for s in common_seasons], marker='o', label='B1', color='blue')
axes2[0,0].plot(common_seasons, [d1_metrics.loc[s, 'home_win'] for s in common_seasons], marker='s', label='D1', color='red')
axes2[0,0].set_title('Home win rate comparison')
axes2[0,0].set_xlabel('Season')
axes2[0,0].set_ylabel('Proportion')
axes2[0,0].legend()
axes2[0,0].grid(True, linestyle='--', alpha=0.6)

# Over 2.5 rate comparison
axes2[0,1].plot(common_seasons, [b1_metrics.loc[s, 'over_rate'] for s in common_seasons], marker='o', label='B1', color='blue')
axes2[0,1].plot(common_seasons, [d1_metrics.loc[s, 'over_rate'] for s in common_seasons], marker='s', label='D1', color='red')
axes2[0,1].set_title('Over 2.5 goals rate')
axes2[0,1].set_xlabel('Season')
axes2[0,1].set_ylabel('Proportion')
axes2[0,1].legend()
axes2[0,1].grid(True, linestyle='--', alpha=0.6)

# Avg goals per match comparison
axes2[1,0].plot(common_seasons, [b1_metrics.loc[s, 'avg_goals'] for s in common_seasons], marker='o', label='B1', color='blue')
axes2[1,0].plot(common_seasons, [d1_metrics.loc[s, 'avg_goals'] for s in common_seasons], marker='s', label='D1', color='red')
axes2[1,0].set_title('Avg total goals per match')
axes2[1,0].set_xlabel('Season')
axes2[1,0].set_ylabel('Goals')
axes2[1,0].legend()
axes2[1,0].grid(True, linestyle='--', alpha=0.6)

# Bar chart: overall averages
overall_b1 = {
    'Home win': b1_metrics['home_win'].mean(),
    'Over 2.5': b1_metrics['over_rate'].mean(),
    'Avg goals': b1_metrics['avg_goals'].mean(),
}
overall_d1 = {
    'Home win': d1_metrics['home_win'].mean(),
    'Over 2.5': d1_metrics['over_rate'].mean(),
    'Avg goals': d1_metrics['avg_goals'].mean(),
}
x = np.arange(3)
width = 0.35
axes2[1,1].bar(x - width/2, [overall_b1['Home win'], overall_b1['Over 2.5'], overall_b1['Avg goals']], width, label='B1', color='blue')
axes2[1,1].bar(x + width/2, [overall_d1['Home win'], overall_d1['Over 2.5'], overall_d1['Avg goals']], width, label='D1', color='red')
axes2[1,1].set_title('Overall averages (2020–2025)')
axes2[1,1].set_xticks(x)
axes2[1,1].set_xticklabels(['Home win', 'Over 2.5', 'Avg goals'])
axes2[1,1].set_ylabel('Proportion / Goals')
axes2[1,1].legend()
axes2[1,1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
compare_path = '/data/.openclaw/workspace/data/football-data/compare_b1_d1.png'
plt.savefig(compare_path, dpi=150)
print(f"Saved comparison plot to {compare_path}")
plt.close()

# Print overall comparison table
print("\nOverall averages (2020–2025):")
print(f"{'Metric':<12} {'B1':<8} {'D1':<8} {'Diff (D1-B1)':<12}")
for key in ['Home win', 'Over 2.5', 'Avg goals']:
    b = overall_b1[key]
    d = overall_d1[key]
    diff = d - b
    print(f"{key:<12} {b:.3f}     {d:.3f}     {diff:+.3f}")