import pandas as pd
import numpy as np
import os

def explore_file(path):
    df = pd.read_csv(path)
    # Parse date (format dd/mm/yyyy)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    print(f"File: {os.path.basename(path)}")
    print(f"  Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Unique seasons: {df['Date'].dt.year.nunique()}")
    # Count matches per season
    for year, count in df['Date'].dt.year.value_counts().sort_index().items():
        print(f"    {year}: {count} matches")
    # Home wins / draws / away wins
    if 'FTR' in df.columns:
        h = (df['FTR'] == 'H').sum()
        d = (df['FTR'] == 'D').sum()
        a = (df['FTR'] == 'A').sum()
        total = h + d + a
        print(f"  Home wins: {h} ({h/total:.1%})")
        print(f"  Draws: {d} ({d/total:.1%})")
        print(f"  Away wins: {a} ({a/total:.1%})")
    # Average goals per match
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        avg_hg = df['FTHG'].mean()
        avg_ag = df['FTAG'].mean()
        print(f"  Avg home goals: {avg_hg:.2f}")
        print(f"  Avg away goals: {avg_ag:.2f}")
        print(f"  Total goals per match: {avg_hg + avg_ag:.2f}")
    # Betting odds columns
    odds_cols = [c for c in df.columns if 'B365' in c]
    print(f"  Bet365 odds columns: {len(odds_cols)}")
    print()

if __name__ == '__main__':
    base = '/data/.openclaw/workspace/data/football-data'
    for f in ['B1.csv', 'D1.csv']:
        explore_file(os.path.join(base, f))
    # Maybe also check B1(4).csv and D1(4).csv (latest seasons?)
    for f in ['B1(4).csv', 'D1(4).csv']:
        if os.path.exists(os.path.join(base, f)):
            explore_file(os.path.join(base, f))
        else:
            print(f"File {f} not downloaded yet")