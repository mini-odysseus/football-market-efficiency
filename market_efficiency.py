import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, glob, sys

def load_league_files(prefix):
    base = '/data/.openclaw/workspace/data/football-data'
    files = sorted(glob.glob(os.path.join(base, f'{prefix}*.csv')))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Season'] = df['Date'].dt.year.astype(str)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

b1 = load_league_files('B1')
d1 = load_league_files('D1')
print(f"B1 matches: {len(b1)}")
print(f"D1 matches: {len(d1)}")

def analyze_market(df, league_name):
    results = {}
    
    # 1X2 market
    odds_cols = ['B365H', 'B365D', 'B365A']
    if all(c in df.columns for c in odds_cols):
        subset = df[odds_cols + ['FTR']].dropna()
        # Implied probabilities (raw)
        subset['ipH'] = 1 / subset['B365H']
        subset['ipD'] = 1 / subset['B365D']
        subset['ipA'] = 1 / subset['B365A']
        # Overround per match
        subset['overround'] = subset['ipH'] + subset['ipD'] + subset['ipA']
        # Normalized implied probabilities
        subset['npH'] = subset['ipH'] / subset['overround']
        subset['npD'] = subset['ipD'] / subset['overround']
        subset['npA'] = subset['ipA'] / subset['overround']
        
        # Actual frequencies
        actual_h = (subset['FTR'] == 'H').mean()
        actual_d = (subset['FTR'] == 'D').mean()
        actual_a = (subset['FTR'] == 'A').mean()
        implied_h = subset['npH'].mean()
        implied_d = subset['npD'].mean()
        implied_a = subset['npA'].mean()
        
        # Expected value (decimal odds)
        ev_h = actual_h - implied_h
        ev_d = actual_d - implied_d
        ev_a = actual_a - implied_a
        
        results['1X2'] = {
            'actual': [actual_h, actual_d, actual_a],
            'implied': [implied_h, implied_d, implied_a],
            'ev': [ev_h, ev_d, ev_a],
            'overround_avg': subset['overround'].mean(),
            'n_matches': len(subset)
        }
        print(f"{league_name} 1X2: H {actual_h:.3f} vs {implied_h:.3f} (ev {ev_h:+.4f}), D {actual_d:.3f} vs {implied_d:.3f} (ev {ev_d:+.4f}), A {actual_a:.3f} vs {implied_a:.3f} (ev {ev_a:+.4f})")
        print(f"  Avg overround: {subset['overround'].mean():.3f}")
    
    # Over/Under 2.5 goals
    if 'B365>2.5' in df.columns and 'B365<2.5' in df.columns:
        subset = df[['B365>2.5', 'B365<2.5']].dropna()
        subset['over_2.5_actual'] = (df['FTHG'] + df['FTAG']) > 2.5
        subset = subset.dropna()
        # Implied probabilities
        subset['ip_over'] = 1 / subset['B365>2.5']
        subset['ip_under'] = 1 / subset['B365<2.5']
        subset['overround_ou'] = subset['ip_over'] + subset['ip_under']
        subset['np_over'] = subset['ip_over'] / subset['overround_ou']
        subset['np_under'] = subset['ip_under'] / subset['overround_ou']
        
        actual_over = subset['over_2.5_actual'].mean()
        actual_under = 1 - actual_over
        implied_over = subset['np_over'].mean()
        implied_under = subset['np_under'].mean()
        ev_over = actual_over - implied_over
        ev_under = actual_under - implied_under
        
        results['OverUnder'] = {
            'actual': [actual_over, actual_under],
            'implied': [implied_over, implied_under],
            'ev': [ev_over, ev_under],
            'overround_avg': subset['overround_ou'].mean(),
            'n_matches': len(subset)
        }
        print(f"{league_name} Over/Under 2.5: Over {actual_over:.3f} vs {implied_over:.3f} (ev {ev_over:+.4f}), Under {actual_under:.3f} vs {implied_under:.3f} (ev {ev_under:+.4f})")
        print(f"  Avg overround: {subset['overround_ou'].mean():.3f}")
    
    # Asian Handicap (simplified: assume AHh = -0.5, 0, +0.5 etc. We'll just examine odds)
    # We'll skip complex mapping for now.
    
    return results

print("\n=== Market Efficiency Analysis ===\n")
b1_res = analyze_market(b1, 'B1')
print()
d1_res = analyze_market(d1, 'D1')

# Plot calibration for 1X2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (league, res) in zip(axes, [('B1', b1_res), ('D1', d1_res)]):
    if '1X2' in res:
        act = res['1X2']['actual']
        imp = res['1X2']['implied']
        labels = ['Home', 'Draw', 'Away']
        x = np.arange(3)
        width = 0.35
        ax.bar(x - width/2, act, width, label='Actual', color='steelblue')
        ax.bar(x + width/2, imp, width, label='Implied', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Probability')
        ax.set_title(f'{league} 1X2 Market Calibration')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        # Add EV as text
        for i, ev in enumerate(res['1X2']['ev']):
            ax.text(i, max(act[i], imp[i]) + 0.02, f'EV {ev:+.3f}', ha='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No 1X2 data', ha='center', va='center')
        ax.set_title(f'{league} 1X2')

plt.tight_layout()
calibration_path = '/data/.openclaw/workspace/data/football-data/calibration_1x2.png'
plt.savefig(calibration_path, dpi=150)
print(f"\nSaved calibration plot to {calibration_path}")

# Over/Under calibration plot
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
for ax, (league, res) in zip(axes2, [('B1', b1_res), ('D1', d1_res)]):
    if 'OverUnder' in res:
        act = res['OverUnder']['actual']
        imp = res['OverUnder']['implied']
        labels = ['Over 2.5', 'Under 2.5']
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, act, width, label='Actual', color='steelblue')
        ax.bar(x + width/2, imp, width, label='Implied', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Probability')
        ax.set_title(f'{league} Over/Under 2.5 Calibration')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        for i, ev in enumerate(res['OverUnder']['ev']):
            ax.text(i, max(act[i], imp[i]) + 0.02, f'EV {ev:+.3f}', ha='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Over/Under data', ha='center', va='center')
        ax.set_title(f'{league} Over/Under')

plt.tight_layout()
ou_cal_path = '/data/.openclaw/workspace/data/football-data/calibration_ou.png'
plt.savefig(ou_cal_path, dpi=150)
print(f"Saved Over/Under calibration plot to {ou_cal_path}")

# Season-by-season efficiency for 1X2 (home win implied vs actual)
def season_efficiency(df, league_name):
    if 'B365H' not in df.columns:
        return
    seasons = sorted(df['Season'].unique())
    home_actual = []
    home_implied = []
    for s in seasons:
        sub = df[df['Season'] == s].copy()
        sub = sub[['B365H', 'B365D', 'B365A', 'FTR']].dropna()
        if len(sub) == 0:
            continue
        # Normalized implied prob for home win
        ipH = 1 / sub['B365H']
        ipD = 1 / sub['B365D']
        ipA = 1 / sub['B365A']
        over = ipH + ipD + ipA
        npH = ipH / over
        home_implied.append(npH.mean())
        home_actual.append((sub['FTR'] == 'H').mean())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seasons[:len(home_actual)], home_actual, marker='o', label='Actual home win %', color='steelblue')
    ax.plot(seasons[:len(home_actual)], home_implied, marker='s', label='Implied home win %', color='coral')
    ax.set_xlabel('Season')
    ax.set_ylabel('Probability')
    ax.set_title(f'{league_name} Home Win: Actual vs Implied by Season')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    path = f'/data/.openclaw/workspace/data/football-data/{league_name}_home_efficiency_season.png'
    plt.savefig(path, dpi=150)
    print(f"Saved {league_name} season home efficiency plot to {path}")
    plt.close()

season_efficiency(b1, 'B1')
season_efficiency(d1, 'D1')

print("\n=== Summary ===")
print("Market efficiency indicates whether actual probability matches implied probability.")
print("Positive EV suggests market undervalues outcome; negative EV overvalues.")
print("Overround indicates built‑in bookmaker margin.")
print("Plots show calibration and season trends.")