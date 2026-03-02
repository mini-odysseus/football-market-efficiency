import pandas as pd, os, glob, sys

def summarize_file(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    seasons = df['Date'].dt.year.unique()
    h = (df['FTR'] == 'H').sum()
    d = (df['FTR'] == 'D').sum()
    a = (df['FTR'] == 'A').sum()
    total = h + d + a
    avg_hg = df['FTHG'].mean()
    avg_ag = df['FTAG'].mean()
    return {
        'file': os.path.basename(path),
        'rows': len(df),
        'min_date': min_date,
        'max_date': max_date,
        'seasons': sorted(seasons),
        'home_win_pct': h / total,
        'draw_pct': d / total,
        'away_win_pct': a / total,
        'avg_home_goals': avg_hg,
        'avg_away_goals': avg_ag,
        'total_goals_per_match': avg_hg + avg_ag,
    }

base = '/data/.openclaw/workspace/data/football-data'
files = glob.glob(os.path.join(base, '*.csv'))
print("Found files:", [os.path.basename(f) for f in files])
print()

results = []
for f in sorted(files):
    try:
        res = summarize_file(f)
        results.append(res)
    except Exception as e:
        print(f"Error with {f}: {e}")

for res in results:
    print(f"{res['file']}:")
    print(f"  Rows: {res['rows']}")
    print(f"  Date range: {res['min_date'].date()} to {res['max_date'].date()}")
    print(f"  Seasons: {res['seasons']}")
    print(f"  Home win %: {res['home_win_pct']:.1%}")
    print(f"  Draw %: {res['draw_pct']:.1%}")
    print(f"  Away win %: {res['away_win_pct']:.1%}")
    print(f"  Avg goals (home/away/total): {res['avg_home_goals']:.2f}/{res['avg_away_goals']:.2f}/{res['total_goals_per_match']:.2f}")
    print()