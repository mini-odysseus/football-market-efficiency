"""
Pythagorean rating walk‑forward test for over/under 2.5 goals market.
Pure probability approach: convert team ratings directly to match probability.
Walk‑forward evaluation across seasons (train on season t, test on t+1).
Compare with Bet365 closing odds.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob, warnings
warnings.filterwarnings('ignore')

def load_league_files(prefix):
    """Load all CSV files for a league."""
    base = '/data/.openclaw/workspace/data/football-data'
    files = sorted([f for f in glob.glob(os.path.join(base, f'{prefix}*.csv'))
                    if not f.endswith('.png')])
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Season'] = df['Date'].dt.year.astype(str)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).copy()

def pythagorean_rating(gf, ga, exponent=1.5):
    """Compute Pythagorean rating for a team."""
    # avoid division by zero
    if gf <= 0: gf = 0.01
    if ga <= 0: ga = 0.01
    return (gf ** exponent) / (gf ** exponent + ga ** exponent)

def compute_season_ratings(season_df, exponent=1.5):
    """Compute offensive and defensive ratings per team for a single season."""
    teams = {}
    # accumulate goals for/against
    for _, row in season_df.iterrows():
        h = row['HomeTeam']
        a = row['AwayTeam']
        hg = row['FTHG']
        ag = row['FTAG']
        
        if h not in teams:
            teams[h] = {'GF': 0, 'GA': 0, 'games': 0}
        if a not in teams:
            teams[a] = {'GF': 0, 'GA': 0, 'games': 0}
        
        teams[h]['GF'] += hg
        teams[h]['GA'] += ag
        teams[h]['games'] += 1
        teams[a]['GF'] += ag
        teams[a]['GA'] += hg
        teams[a]['games'] += 1
    
    # compute per‑game averages and rating
    ratings = {}
    for team, stats in teams.items():
        gf_pg = stats['GF'] / stats['games']
        ga_pg = stats['GA'] / stats['games']
        rating = pythagorean_rating(gf_pg, ga_pg, exponent)
        ratings[team] = {'rating': rating, 'gf_pg': gf_pg, 'ga_pg': ga_pg}
    return ratings

def match_probability(home_rating, away_rating, method='additive', league_avg_goals=2.7):
    """
    Convert team ratings to a probability of over 2.5 goals.
    
    Parameters:
    - home_rating, away_rating: Pythagorean ratings (0‑1 scale)
    - method: 'additive' or 'multiplicative'
    - league_avg_goals: average total goals per match in the league
    
    Returns:
    - p_over: predicted probability of over 2.5 goals
    """
    if method == 'additive':
        # Linear scaling: rating sum -> expected goals
        # Assume rating sum of 1.0 corresponds to league average goals
        expected_goals = league_avg_goals * (home_rating + away_rating)
    else:  # multiplicative
        expected_goals = league_avg_goals * home_rating * away_rating * 2
    
    # Convert expected goals to over‑probability via Poisson
    # Simple approximation: p_over = 1 - Poisson(≤2 goals)
    p_under = np.exp(-expected_goals) * (1 + expected_goals + expected_goals**2/2)
    p_over = 1 - p_under
    # Clip to reasonable range
    return np.clip(p_over, 0.05, 0.95)

def walk_forward_evaluation(df, league_name, exponent=1.5, method='additive'):
    """
    Perform walk‑forward test: train on season t, test on t+1.
    
    Returns:
    - results_df: DataFrame with predictions, actuals, Bet365 odds
    - metrics: dict of performance metrics
    """
    seasons = sorted(df['Season'].unique())
    all_rows = []
    
    for i in range(len(seasons) - 1):
        train_season = seasons[i]
        test_season = seasons[i + 1]
        
        print(f"  {train_season} → {test_season}")
        
        train_df = df[df['Season'] == train_season].copy()
        test_df = df[df['Season'] == test_season].copy()
        
        # Compute ratings using training data only
        train_ratings = compute_season_ratings(train_df, exponent)
        
        # Compute league average total goals from training season
        league_avg = train_df['FTHG'].sum() + train_df['FTAG'].sum()
        league_avg /= len(train_df)
        
        # Process each test match
        for _, row in test_df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            home_info = train_ratings.get(home, {'rating': 0.5, 'gf_pg': 1.0, 'ga_pg': 1.0})
            away_info = train_ratings.get(away, {'rating': 0.5, 'gf_pg': 1.0, 'ga_pg': 1.0})
            
            p_over = match_probability(
                home_info['rating'],
                away_info['rating'],
                method=method,
                league_avg_goals=league_avg
            )
            
            total_goals = row['FTHG'] + row['FTAG']
            actual_over = 1 if total_goals > 2.5 else 0
            
            # Bet365 implied probabilities (normalized)
            if 'B365>2.5' in row and pd.notna(row['B365>2.5']):
                odds_over = row['B365>2.5']
                odds_under = row['B365<2.5'] if 'B365<2.5' in row and pd.notna(row['B365<2.5']) else 0
                if odds_over > 0 and odds_under > 0:
                    imp_over = 1 / odds_over
                    imp_under = 1 / odds_under
                    overround = imp_over + imp_under
                    b365_over = imp_over / overround
                    b365_under = imp_under / overround
                else:
                    b365_over = b365_under = None
            else:
                b365_over = b365_under = None
            
            all_rows.append({
                'season': test_season,
                'date': row['Date'],
                'home': home,
                'away': away,
                'home_rating': home_info['rating'],
                'away_rating': away_info['rating'],
                'total_goals': total_goals,
                'actual_over': actual_over,
                'pred_over': p_over,
                'b365_over': b365_over,
                'b365_under': b365_under,
                'odds_over': odds_over if 'odds_over' in locals() else None,
                'odds_under': odds_under if 'odds_under' in locals() else None,
            })
    
    results = pd.DataFrame(all_rows)
    
    # Calculate metrics
    if len(results) == 0:
        return results, {}
    
    # Filter rows where we have both prediction and Bet365 odds
    valid = results.dropna(subset=['pred_over', 'actual_over', 'b365_over'])
    if len(valid) == 0:
        return results, {}
    
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    y_true = valid['actual_over']
    y_pred = valid['pred_over']
    y_b365 = valid['b365_over']
    
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    brier = brier_score_loss(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    brier_b365 = brier_score_loss(y_true, y_b365)
    logloss_b365 = log_loss(y_true, y_b365)
    
    # Profit simulation (unit bet on model if pred > b365 implied)
    valid['bet_model'] = valid['pred_over'] > valid['b365_over']
    valid['stake'] = 1
    valid['odds_used'] = valid.apply(
        lambda r: 1/r['b365_over'] if r['bet_model'] else 1/r['b365_under'], axis=1)
    valid['won'] = valid.apply(
        lambda r: (r['actual_over'] == 1 and r['bet_model']) or
                  (r['actual_over'] == 0 and not r['bet_model']), axis=1)
    valid['profit'] = valid.apply(
        lambda r: (r['odds_used'] - 1) if r['won'] else -1, axis=1)
    total_profit = valid['profit'].sum()
    total_bets = len(valid)
    roi = total_profit / total_bets if total_bets > 0 else 0
    
    metrics = {
        'n_matches': len(valid),
        'auc': auc,
        'brier': brier,
        'logloss': logloss,
        'brier_b365': brier_b365,
        'logloss_b365': logloss_b365,
        'total_profit': total_profit,
        'total_bets': total_bets,
        'roi': roi,
        'brier_diff': brier - brier_b365,
        'logloss_diff': logloss - logloss_b365,
    }
    return results, metrics

def run_full_analysis():
    print("Loading data...")
    b1 = load_league_files('B1')
    d1 = load_league_files('D1')
    print(f"B1: {len(b1)} matches, seasons {sorted(b1['Season'].unique())}")
    print(f"D1: {len(d1)} matches, seasons {sorted(d1['Season'].unique())}")
    
    exponents = [1.0, 1.5, 2.0, 2.5]
    methods = ['additive', 'multiplicative']
    
    summary = []
    
    for league_name, df in [('B1', b1), ('D1', d1)]:
        print(f"\n{'='*60}")
        print(f"League: {league_name}")
        print(f"{'='*60}")
        
        for exp in exponents:
            for method in methods:
                print(f"\nExponent {exp}, method {method}")
                results, metrics = walk_forward_evaluation(df, league_name, exp, method)
                
                if metrics:
                    summary.append({
                        'league': league_name,
                        'exponent': exp,
                        'method': method,
                        **metrics
                    })
                    
                    print(f"  Matches: {metrics['n_matches']}")
                    print(f"  AUC: {metrics['auc']:.4f}")
                    print(f"  Brier: {metrics['brier']:.4f} (Bet365: {metrics['brier_b365']:.4f})")
                    print(f"  LogLoss: {metrics['logloss']:.4f} (Bet365: {metrics['logloss_b365']:.4f})")
                    print(f"  ROI: {metrics['roi']:.3f} (profit {metrics['total_profit']:.1f} units)")
                    if metrics['brier_diff'] < 0:
                        print(f"  → Model better than Bet365 (Brier diff {metrics['brier_diff']:.4f})")
                    else:
                        print(f"  → Bet365 better than model (Brier diff {metrics['brier_diff']:.4f})")
                else:
                    print("  Not enough valid matches.")
    
    # Save summary table
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = '/data/.openclaw/workspace/data/football-data/pythagorean_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        
        # Best configuration per league
        for league in ['B1', 'D1']:
            sub = summary_df[summary_df['league'] == league]
            if len(sub) > 0:
                best = sub.loc[sub['brier_diff'].idxmin()]
                print(f"\nBest for {league}: exponent {best['exponent']}, method {best['method']}")
                print(f"  Brier diff: {best['brier_diff']:.4f}, ROI: {best['roi']:.3f}")
    
    # Run detailed analysis with best configuration (exponent 1.5, additive)
    print("\n" + "="*60)
    print("Detailed walk‑forward results (exponent=1.5, method=additive)")
    print("="*60)
    
    for league_name, df in [('B1', b1), ('D1', d1)]:
        print(f"\nLeague: {league_name}")
        results, metrics = walk_forward_evaluation(df, league_name, exponent=1.5, method='additive')
        
        if len(results) > 0:
            out_path = f'/data/.openclaw/workspace/data/football-data/pythagorean_{league_name}_detailed.csv'
            results.to_csv(out_path, index=False)
            print(f"  Detailed results saved to {out_path}")
            
            # Generate calibration plot
            fig, ax = plt.subplots(figsize=(6,5))
            from sklearn.calibration import calibration_curve
            y_true = results.dropna(subset=['pred_over', 'actual_over'])['actual_over']
            y_pred = results.dropna(subset=['pred_over', 'actual_over'])['pred_over']
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
            ax.plot([0,1], [0,1], 'k--', label='Perfect')
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('Actual frequency')
            ax.set_title(f'{league_name}: Pythagorean model calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
            cal_path = f'/data/.openclaw/workspace/data/football-data/pythagorean_calibration_{league_name}.png'
            plt.tight_layout()
            plt.savefig(cal_path, dpi=150)
            plt.close()
            print(f"  Calibration plot saved to {cal_path}")
            
            # Profit curve
            if 'profit' in results.columns:
                results_valid = results.dropna(subset=['profit'])
                if len(results_valid) > 0:
                    fig, ax = plt.subplots(figsize=(6,5))
                    cum_profit = results_valid['profit'].cumsum()
                    ax.plot(cum_profit.index, cum_profit, linewidth=2)
                    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Bet number')
                    ax.set_ylabel('Cumulative profit (units)')
                    ax.set_title(f'{league_name}: Cumulative profit from Pythagorean model')
                    ax.grid(True, alpha=0.3)
                    profit_path = f'/data/.openclaw/workspace/data/football-data/pythagorean_profit_{league_name}.png'
                    plt.tight_layout()
                    plt.savefig(profit_path, dpi=150)
                    plt.close()
                    print(f"  Profit curve saved to {profit_path}")
    
    print("\nAnalysis complete. Results are in /data/.openclaw/workspace/data/football-data/")

if __name__ == '__main__':
    run_full_analysis()