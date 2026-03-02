# Football Betting Market Efficiency Analysis

Analysis of football betting market efficiency using historical match data and Bet365 closing odds.

## Overview

This repository contains Python scripts used to analyze the efficiency of football betting markets (specifically the over/under 2.5 goals and 1X2 markets) using historical match data from the Belgian First Division A (B1) and German Bundesliga (D1).

The analysis includes:
- Market calibration tests (actual vs implied probabilities)
- Expected value (EV) calculations
- Seasonal efficiency trends
- Walk‑forward predictive modeling
- Profit/loss simulations against Bet365 closing odds

## Key Findings

### 1. Market Efficiency Levels
- **1X2 Market**: Generally efficient, except Bundesliga draws are significantly undervalued (EV +1.86%)
- **Over/Under 2.5 Goals**: Bundesliga market consistently undervalued over 2.5 goals (EV +1.54%); Belgian league recently overvalued
- **Bookmaker Margins**: 5–7% overround (lower in Bundesliga)

### 2. Predictive Modeling Failure
- Logistic regression model trained on 2020‑2022 Belgian league data
- Tested on 2023‑2025 data (walk‑forward validation)
- No profitable threshold found – model lost money across nearly all settings
- Conclusion: Market closing odds are efficient; simple team‑form features don't provide edge

### 3. League Differences
- **Bundesliga**: Higher‑scoring, more volatile home advantage, undervalued draws
- **Belgian League**: Declining goal frequency, market adjusted to overvalue over 2.5 in recent seasons

## Repository Structure

| File | Purpose |
|------|---------|
| `market_efficiency.py` | Main calibration and EV calculations for 1X2 and over/under markets |
| `walkforward_model.py` | Logistic regression model with temporal split and betting simulation |
| `ou_consistency.py` | Seasonal EV analysis with statistical significance testing |
| `d1_analysis.py` | Bundesliga‑specific analysis and visualization |
| `explore.py` / `summarize.py` | Dataset exploration and summary statistics |
| `model_walkforward.py` | Earlier model iteration (cross‑league testing) |

## Data Source

Football‑data CSV files (Bet365‑style format) containing:
- Match results (goals, shots, cards, etc.)
- Bet365 closing odds (1X2, over/under, Asian handicap, etc.)
- Seasons: 2020‑2025 (B1), 2020‑2026 (D1)

## Usage

1. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn scipy
   ```

2. **Run market efficiency analysis**:
   ```python
   python market_efficiency.py
   ```

3. **Run walk‑forward model**:
   ```python
   python walkforward_model.py
   ```

## Dependencies

- Python 3.13+
- pandas, numpy
- matplotlib
- scikit‑learn
- scipy (for binomial tests)

## Results Visualization

The analysis generates several PNG charts:
- `calibration_1x2.png` – 1X2 market calibration
- `calibration_ou.png` – Over/under market calibration  
- `ou_ev_by_season.png` – Seasonal EV trends
- `walkforward_profit.png` – P&L chart from model betting

## Learnings & Reflections

Detailed reflections are documented in the Obsidian vault (see `Journal/2026-03-02-reflection-market-efficiency-testing.md`).

Key takeaways:
- Market efficiency is high for closing odds
- Walk‑forward validation essential to avoid overfitting
- Feature engineering without private information unlikely to beat market
- Bundesliga draws present possible edge (requires further testing)

## License

MIT