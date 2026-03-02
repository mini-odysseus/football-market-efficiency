import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def load_league_files(prefix):
    """Load all CSV files for a league prefix (B1, D1, etc.)"""
    base = '/data/.openclaw/workspace/data/football-data'
    files = sorted(glob.glob(os.path.join(base, f'{prefix}*.csv')))
    dfs = []
    for f in files:
        if f.endswith('.png'):  # skip images
            continue
        df = pd.read_csv(f)
        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        # Assign season: year of date (could refine to Aug-May seasonal split)
        df['Season'] = df['Date'].dt.year.astype(str)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_pythagorean_ratings(df, exponent=1.5):
    """
    Compute Pythagorean ratings for each team in each season.
    
    Parameters:
    - df: DataFrame with columns ['Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    - exponent: exponent for Pythagorean formula (default 1.5)
    
    Returns:
    - dict: {season: {team: rating}}
    """
    ratings = {}
    seasons = df['Season'].unique()
    
    for season in seasons:
        season_df = df[df['Season'] == season].copy()
        team_stats = {}
        
        # Collect goals for and against for each team (home and away)
        for _, row in season_df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            hg = row['FTHG']
            ag = row['FTAG']
            
            # Home team: goals for = hg, goals against = ag
            if home not in team_stats:
                team_stats[home] = {'GF': 0, 'GA': 0, 'games': 0}
            team_stats[home]['GF'] += hg
            team_stats[home]['GA'] += ag
            team_stats[home]['games'] += 1
            
            # Away team: goals for = ag, goals against = hg
            if away not in team_stats:
                team_stats[away] = {'GF': 0, 'GA': 0, 'games': 0}
            team_stats[away]['GF'] += ag
            team_stats[away]['GA'] += hg
            team_stats[away]['games'] += 1
        
        # Compute Pythagorean rating for each team
        season_ratings = {}
        for team, stats in team_stats.items():
            gf_per_game = stats['GF'] / stats['games']
            ga_per_game = stats['GA'] / stats['games']
            # Avoid division by zero or negative
            if gf_per_game <= 0:
                gf_per_game = 0.01
            if ga_per_game <= 0:
                ga_per_game = 0.01
            # Pythagorean rating
            rating = (gf_per_game ** exponent) / (gf_per_game ** exponent + ga_per_game ** exponent)
            season_ratings[team] = rating
        
        ratings[season] = season_ratings
    
    return ratings

def create_features(df, ratings_dict):
    """
    Create feature matrix X and target y for over/under 2.5 goals.
    
    Parameters:
    - df: DataFrame with match data
    - ratings_dict: output from compute_pythagorean_ratings
    
    Returns:
    - X: DataFrame with features
    - y: Series (1 for over 2.5, 0 for under)
    """
    X_rows = []
    y_rows = []
    
    for _, row in df.iterrows():
        season = row['Season']
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Get ratings for this season
        season_ratings = ratings_dict.get(season, {})
        home_rating = season_ratings.get(home, 0.5)  # default 0.5 if missing
        away_rating = season_ratings.get(away, 0.5)
        
        # Features
        rating_diff = home_rating - away_rating
        total_rating = home_rating + away_rating
        
        X_rows.append([home_rating, away_rating, rating_diff, total_rating])
        
        # Target: over 2.5 goals
        total_goals = row['FTHG'] + row['FTAG']
        y_rows.append(1 if total_goals > 2.5 else 0)
    
    X = pd.DataFrame(X_rows, columns=['home_rating', 'away_rating', 'rating_diff', 'total_rating'])
    y = pd.Series(y_rows, name='over_2.5')
    return X, y

def walk_forward_evaluation(df, exponent=1.5):
    """
    Perform walk-forward evaluation: train on season t, test on season t+1.
    
    Parameters:
    - df: DataFrame with all matches
    - exponent: Pythagorean exponent
    
    Returns:
    - results_df: DataFrame with predictions and actual outcomes
    - metrics: dict of performance metrics
    """
    seasons = sorted(df['Season'].unique())
    
    all_predictions = []
    
    for i in range(len(seasons) - 1):
        train_season = seasons[i]
        test_season = seasons[i + 1]
        
        print(f"Training on {train_season}, testing on {test_season}")
        
        # Split data
        train_df = df[df['Season'] == train_season].copy()
        test_df = df[df['Season'] == test_season].copy()
        
        # Compute ratings using training data only
        train_ratings = compute_pythagorean_ratings(train_df, exponent=exponent)
        
        # Create features for training and testing
        X_train, y_train = create_features(train_df, train_ratings)
        X_test, y_test = create_features(test_df, train_ratings)  # use same ratings (from training season)
        
        # Train logistic regression
        if len(X_train) > 0 and len(np.unique(y_train)) > 1:
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            
            # Predict probabilities on test set
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Collect results
            for idx, (_, row) in enumerate(test_df.iterrows()):
                all_predictions.append({
                    'Season': test_season,
                    'Date': row['Date'],
                    'HomeTeam': row['HomeTeam'],
                    'AwayTeam': row['AwayTeam'],
                    'FTHG': row['FTHG'],
                    'FTAG': row['FTAG'],
                    'total_goals': row['FTHG'] + row['FTAG'],
                    'over_2.5_actual': 1 if row['FTHG'] + row['FTAG'] > 2.5 else 0,
                    'over_2.5_pred': y_pred_proba[idx],
                    'B365_over': 1/row['B365>2.5'] if 'B365>2.5' in row and not pd.isna(row['B365>2.5']) else None,
                    'B365_under': 1/row['B365<2.5'] if 'B365<2.5' in row and not pd.isna(row['B365<2.5']) else None
                })
    
    results_df = pd.DataFrame(all_predictions)
    
    # Compute metrics
    if len(results_df) > 0:
        results_df = results_df.dropna(subset=['over_2.5_pred', 'over_2.5_actual'])
        y_true = results_df['over_2.5_actual']
        y_pred = results_df['over_2.5_pred']
        
        # AUC
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        # Brier score
        brier = brier_score_loss(y_true, y_pred)
        # Log loss
        logloss = log_loss(y_true, y_pred)
        
        # Compare with Bet365 odds
        if 'B365_over' in results_df.columns:
            # Filter rows with valid odds
            odds_df = results_df.dropna(subset=['B365_over', 'B365_under'])
            if len(odds_df) > 0:
                # Normalize implied probabilities (remove overround)
                odds_df['overround'] = odds_df['B365_over'] + odds_df['B365_under']
                odds_df['B365_over_norm'] = odds_df['B365_over'] / odds_df['overround']
                # Brier score for Bet365
                brier_b365 = brier_score_loss(odds_df['over_2.5_actual'], odds_df['B365_over_norm'])
                # Log loss for Bet365
                logloss_b365 = log_loss(odds_df['over_2.5_actual'], odds_df['B365_over_norm'])
            else:
                brier_b365 = None
                logloss_b365 = None
        else:
            brier_b365 = None
            logloss_b365 = None
        
        metrics = {
            'n_matches': len(results_df),
            'auc': auc,
            'brier': brier,
            'logloss': logloss,
            'brier_b365': brier_b365,
            'logloss_b365': logloss_b365
        }
    else:
        metrics = {}
    
    return results_df, metrics

def main():
    # Load data for B1 and D1
    print("Loading B1 data...")
    b1 = load_league_files('B1')
    print(f"B1 matches: {len(b1)}")
    print("Loading D1 data...")
    d1 = load_league_files('D1')
    print(f"D1 matches: {len(d1)}")
    
    # Test different exponents
    exponents = [1.0, 1.5, 2.0, 2.5]
    
    for league_name, df in [('B1', b1), ('D1', d1)]:
        print(f"\n{'='*50}")
        print(f"League: {league_name}")
        print(f"{'='*50}")
        
        for exp in exponents:
            print(f"\nExponent: {exp}")
            results, metrics = walk_forward_evaluation(df, exponent=exp)
            
            if metrics:
                print(f"  Matches: {metrics['n_matches']}")
                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  Brier: {metrics['brier']:.4f}")
                print(f"  LogLoss: {metrics['logloss']:.4f}")
                if metrics['brier_b365']:
                    print(f"  Brier (Bet365): {metrics['brier_b365']:.4f}")
                    print(f"  LogLoss (Bet365): {metrics['logloss_b365']:.4f}")
                    # Compare: lower is better
                    brier_diff = metrics['brier'] - metrics['brier_b365']
                    print(f"  Brier diff (model - Bet365): {brier_diff:.4f}")
                    if brier_diff < 0:
                        print(f"    -> Model better than Bet365")
                    else:
                        print(f"    -> Bet365 better than model")
    
    # Save detailed results for the best exponent
    print("\nSaving detailed results for exponent 1.5...")
    results_b1, _ = walk_forward_evaluation(b1, exponent=1.5)
    results_d1, _ = walk_forward_evaluation(d1, exponent=1.5)
    
    results_b1.to_csv('/data/.openclaw/workspace/data/football-data/pythagorean_b1_results.csv', index=False)
    results_d1.to_csv('/data/.openclaw/workspace/data/football-data/pythagorean_d1_results.csv', index=False)
    print("Results saved to CSV files.")
    
    # Generate plots
    generate_plots(results_b1, results_d1)

def generate_plots(results_b1, results_d1):
    """Generate calibration plots and profit curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Calibration plot for B1
    ax = axes[0, 0]
    plot_calibration(results_b1, ax, title='B1: Pythagorean Model Calibration')
    
    # Calibration plot for D1
    ax = axes[0, 1]
    plot_calibration(results_d1, ax, title='D1: Pythagorean Model Calibration')
    
    # Profit curve for B1
    ax = axes[1, 0]
    plot_profit_curve(results_b1, ax, title='B1: Profit Curve vs Bet365')
    
    # Profit curve for D1
    ax = axes[1, 1]
    plot_profit_curve(results_d1, ax, title='D1: Profit Curve vs Bet365')
    
    plt.tight_layout()
    plt.savefig('/data/.openclaw/workspace/data/football-data/pythagorean_analysis.png', dpi=150)
    print("Plot saved: pythagorean_analysis.png")

def plot_calibration(results_df, ax, title='Calibration Plot'):
    """Plot reliability diagram."""
    from sklearn.calibration import calibration_curve
    y_true = results_df['over_2.5_actual']
    y_pred = results_df['over_2.5_pred']
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_profit_curve(results_df, ax, title='Profit Curve'):
    """Plot cumulative profit from betting on model predictions vs Bet365."""
    if 'B365_over_norm' not in results_df.columns:
        ax.text(0.5, 0.5, 'Bet365 odds not available', ha='center', va='center')
        ax.set_title(title)
        return
    
    # Sort matches by model prediction (descending confidence in over)
    df_sorted = results_df.sort_values('over_2.5_pred', ascending=False).reset_index(drop=True)
    
    # Betting strategy: bet on over when model probability > implied probability + threshold
    # Simple approach: bet on over when model prob > Bet365 implied prob
    df_sorted['bet_over'] = df_sorted['over_2.5_pred'] > df_sorted['B365_over_norm']
    df_sorted['stake'] = 1  # unit bet
    df_sorted['odds'] = df_sorted.apply(lambda row: 1/row['B365_over_norm'] if row['bet_over'] else 1/row['B365_under_norm'], axis=1)
    df_sorted['won'] = df_sorted.apply(lambda row: 
        (row['over_2.5_actual'] == 1 and row['bet_over']) or 
        (row['over_2.5_actual'] == 0 and not row['bet_over']), axis=1)
    df_sorted['profit'] = df_sorted.apply(lambda row: (row['odds'] - 1) if row['won'] else -1, axis=1)
    
    # Cumulative profit
    df_sorted['cumulative_profit'] = df_sorted['profit'].cumsum()
    
    ax.plot(df_sorted.index, df_sorted['cumulative_profit'], linewidth=2, label='Model')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Number of Bets')
    ax.set_ylabel('Cumulative Profit (units)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

if __name__ == '__main__':
    main()