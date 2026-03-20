# P2-ETF-MERTON-ANN

**Merton Optimal Portfolio with Semi-Markov Regime Switching and ANN Feedback Control**

Daily next-day ETF signals for equity and fixed income universes, using a neural network trained to solve Merton's intertemporal portfolio problem under regime-aware market dynamics.

---

## Research Foundation

**Primary Paper**

> *"Intertemporal Optimal Portfolio Allocation under Regime Switching Using Artificial Neural Networks"*
> Uri Carl, Yaacov Kopeliovich, Michael Pokojovy, Kevin Shea
> University of Connecticut / Old Dominion University / Blue Frontiers Partners / Disciplined Alpha
> February 2025
> [Paper PDF](https://finance-business.media.uconn.edu/wp-content/uploads/sites/723/2025/02/Michael-Polojovy-Paper.pdf)

The core insight: rather than deriving and solving the Hamilton-Jacobi-Bellman PDE (the classical approach to Merton's problem), a small ANN can be trained directly to represent the optimal feedback control. The ANN is trained on synthetic GBM paths calibrated to historical regime statistics — it never sees actual historical returns during training, eliminating the overfitting risk inherent in data-driven approaches.

**Main finding:** *"Regimes count."* A regime-aware ANN allocation clearly outperforms the classical regime-agnostic Merton portfolio on both synthetic and 18-year historical backtests.

---

## Our Approach — Key Differences from the Paper

The paper uses monthly returns on 12 institutional indices (Russell, Barclays, HFRI) sourced from FactSet, with VIX as the sole regime indicator. We adapt the framework for daily ETF trading with several improvements:

| Dimension | Paper | This Project |
|---|---|---|
| Universe | 12 institutional indices + cash | 15 equity ETFs / 13 FI ETFs |
| Frequency | Monthly | Daily |
| Regime indicator | VIX only (all assets) | VIX for equity, MOVE for fixed income |
| Calibration window | Rolling 17 years | Rolling — configurable |
| Output | Full weight vector | Single ETF (winner-takes-all argmax) |
| Training frequency | On-demand | Daily retraining after market close |
| Data source | FactSet (proprietary) | Yahoo Finance + FRED (free) |

**The key architectural decision — MOVE index for fixed income:**
The paper uses VIX for all assets including bonds. VIX measures equity option implied volatility and is a poor regime indicator for fixed income. We use the ICE BofAML MOVE Index (`^MOVE`) as the regime barometer for the fixed income module, which directly measures US Treasury option implied volatility and responds immediately to rate uncertainty, curve stress, and Fed policy shifts.

---

## How It Works

### Step 1 — Regime Detection

For each module, compute the 12-month geometric moving average of the regime indicator:

```
MA_t = exp( (1/252) x sum( log(VIX_{t-i}) ) )   [equity]
MA_t = exp( (1/252) x sum( log(MOVE_{t-i}) ) )  [fixed income]
```

K-means clustering (K=2) on the log-transformed moving averages produces a threshold separating **risk-on** (low vol, invest) from **risk-off** (high vol, defensive) regimes. The semi-Markov property captures the empirically observed long holding times per regime (~4.5 years average), which standard HMM cannot model.

### Step 2 — Market Calibration

For each regime separately, estimate from historical ETF returns:
- **mu(regime)** — mean return vector per ETF
- **Sigma(regime)** — covariance matrix across ETFs
- **r(regime)** — risk-free rate (DTB3 from FRED)

### Step 3 — Synthetic Data Generation

Using calibrated parameters, simulate 10,000+ synthetic portfolio paths under a multi-asset GBM with semi-Markov regime switching. The ANN trains on synthetic paths — never on historical returns directly.

### Step 4 — ANN Training (daily)

A small MLP with one hidden layer (~50 parameters) trained via SGD to maximise expected isoelastic utility of terminal wealth:

```
U(W) = W^(1-eta) / (1-eta)    [eta != 1]
U(W) = log(W)                  [eta = 1]
```

**Inputs:** (t/T, log(W/W0), regime)
**Outputs:** Portfolio weights — softmax-normalised, long-only, sum to 1

### Step 5 — Signal Generation

Winner-takes-all argmax selects the single highest-weighted ETF. The next NYSE trading date is computed via `pandas_market_calendars` to ensure the UI always shows the correct hold date accounting for weekends and holidays.

### Daily Pipeline

```
21:30 UTC  ->  daily_data_update.py    append new trading day to HF dataset
22:00 UTC  ->  train_predict.py        recalibrate + retrain ANN + predict
                    |
                    v
               Results pushed to HF dataset
                    |
                    v
               Streamlit dashboard updates automatically
```

---

## ETF Universe

### Option A — Equity (Benchmark: SPY, Regime: ^VIX)

| Ticker | Description |
|---|---|
| QQQ | NASDAQ 100 |
| IWM | Russell 2000 Small Cap |
| IWF | Russell 1000 Growth |
| IWD | Russell 1000 Value |
| XLK | Technology |
| XLF | Financials |
| XLE | Energy |
| XLV | Health Care |
| XLI | Industrials |
| XLY | Consumer Discretionary |
| XLP | Consumer Staples |
| XLU | Utilities |
| GDX | Gold Miners |
| EEM | Emerging Markets |
| EFA | Developed International |

### Option B — Fixed Income + Real Assets (Benchmark: AGG, Regime: ^MOVE)

| Ticker | Description |
|---|---|
| TLT | 20Y+ US Treasury |
| IEF | 7-10Y US Treasury |
| SHY | 1-3Y US Treasury (near-cash) |
| LQD | IG Corporate Bonds |
| HYG | High Yield Corporate Bonds |
| MBB | Mortgage-Backed Securities |
| PFF | Preferred Stock |
| TIP | TIPS (Inflation-Linked) |
| GLD | Gold |
| SLV | Silver |
| VNQ | US REITs |
| DBC | Broad Commodities |
| EMB | Emerging Market Bonds |

### Macro Indicators (FRED — both modules)

| Series | Description |
|---|---|
| DTB3 | 3M T-Bill Rate — risk-free rate |
| DGS10 | 10Y Treasury Yield |
| T10Y2Y | 10Y-2Y Yield Spread |
| T10Y3M | 10Y-3M Yield Spread |
| BAMLH0A0HYM2 | HY Credit Spread |
| BAMLC0A0CM | IG Credit Spread |
| DCOILWTICO | WTI Crude Oil |
| DTWEXBGS | USD Broad Dollar Index |
| T10YIE | 10Y Breakeven Inflation |

---

## Infrastructure

```
READ  data from:  P2SAMAPA/p2-etf-merton-ann-data
WRITE results to: P2SAMAPA/p2-etf-merton-ann-data
                  +-- data/equity.parquet
                  +-- data/fixed_income.parquet
                  +-- signals/equity_signal.json
                  +-- signals/fi_signal.json
                  +-- signals/equity_history.json
                  +-- signals/fi_history.json
DISPLAY at:       Streamlit.io
TRAINING on:      GitHub Actions free tier (CPU, <1 min — ANN is ~50 params)
```

---

## Repository Structure

```
P2-ETF-MERTON-ANN/
|
+-- .github/workflows/
|   +-- seed.yml                  # ONE-TIME: seed full history
|   +-- daily_data_update.yml     # DAILY 21:30 UTC: append new day
|   +-- train_predict.yml         # DAILY 22:00 UTC: retrain + predict
|
+-- config.py                     # ETF universes, FRED series, HF config
+-- seed.py                       # One-time full history seed
+-- daily_data_update.py          # Incremental daily data append
+-- regime_detection.py           # VIX/MOVE geometric MA + K-means
+-- calibration.py                # Estimate mu, Sigma, r per regime
+-- simulation.py                 # Synthetic GBM path generation
+-- ann_model.py                  # MLP ANN feedback controller
+-- train_predict.py              # Daily: calibrate + simulate + train + predict
+-- app.py                        # Streamlit dashboard
+-- requirements.txt
```

---

## GitHub Secrets Required

| Secret | Value |
|---|---|
| `HF_TOKEN` | HuggingFace write token |
| `FRED_API_KEY` | FRED API key (free at fred.stlouisfed.org) |
| `HF_DATASET_REPO` | `P2SAMAPA/p2-etf-merton-ann-data` |

---

## Getting Started

**1. Seed the dataset (run once)**
Actions -> Seed Historical Data -> Run workflow (~10-15 minutes)

**2. Run first training + prediction**
Actions -> Daily Train + Predict -> Run workflow

**3. View the dashboard**
Open the Streamlit app linked in the repo description.

---

## Important Caveats

- Research implementation — not financial advice
- Past backtest performance does not guarantee future results
- GBM assumption underestimates fat tails in actual ETF returns
- Single ETF concentration creates idiosyncratic risk — treat as a signal, not a standalone strategy

---

## Citation

```bibtex
@techreport{carl2025merton,
  title       = {Intertemporal Optimal Portfolio Allocation under Regime
                 Switching Using Artificial Neural Networks},
  author      = {Carl, Uri and Kopeliovich, Yaacov and Pokojovy, Michael
                 and Shea, Kevin},
  institution = {University of Connecticut},
  year        = {2025},
  month       = {February},
  url         = {https://finance-business.media.uconn.edu/wp-content/uploads/
                 sites/723/2025/02/Michael-Polojovy-Paper.pdf}
}
```
