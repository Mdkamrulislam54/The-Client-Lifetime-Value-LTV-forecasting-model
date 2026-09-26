# Client Lifetime Value (LTV) Forecasting Model

A machine learning model that forecasts each client's future commission contribution, enabling financial organizations to prioritize relationship management resources and service levels by client value.

## Overview

Not all clients are worth the same to a brokerage or financial services firm — but without a forward-looking view, RMs and management often allocate attention based on gut feel or current account size alone. This project trains a regression model on historical trading behavior to **forecast future commission generation per client**, then segments clients into **Bronze / Silver / Gold / Platinum** tiers so resources can be allocated where they'll have the most impact.

**Pipeline:**

```
Raw Transactions (Excel)
        │
        ▼
Historical / Future Time Split (cutoff date)
        │
        ▼
Client-Level Feature Engineering (historical window)
        │
        ▼
Target Definition: Future Commission (future window)
        │
        ▼
Gradient Boosting Regressor
        │
        ▼
LTV Forecast per Client
        │
        ▼
Tiering (Bronze / Silver / Gold / Platinum)
        │
        ▼
Ranked LTV Tier List (CSV)
```

## Repository Structure

```
├── The_Client_Lifetime_Value__LTV__forecasting_model.ipynb   # Main analysis & modeling notebook
├── Financial Dataset.xlsx                                     # Raw transaction-level input data
├── Life time value_tiers.csv                                  # Generated output: client LTV forecasts & tiers
├── requirements.txt                                            # Python dependencies
└── README.md                                                   # Project documentation
```

## Data

| File | Description |
|---|---|
| [`Financial Dataset.xlsx`](Financial%20Dataset.xlsx) | Raw transaction-level records per client, including trading dates, turnover, commission, equity, and deposits. |
| [`Life time value_tiers.csv`](Life%20time%20value_tiers.csv) | Model output — every client's forecasted LTV, sorted descending, with assigned value tier. |

## Methodology

### 1. Time-Based Data Split
Rather than a random train/test split, this model uses a **temporal cutoff (2023-04-01)** to separate:
- **Historical window** — used to engineer features describing past client behavior.
- **Future window** — used to compute the actual future commission each client generated, which becomes the prediction target.

This mirrors the real-world forecasting problem: predict what a client will be worth *going forward*, using only information available *up to today*.

### 2. Feature Engineering
Historical transactions are aggregated per client into behavioral features:

| Feature | Description |
|---|---|
| `turnover_h` | Total trading turnover in the historical window |
| `comm_h` | Total commission generated in the historical window |
| `equity_h` | Most recent reported account equity |
| `deposits_h` | Total deposits in the historical window |
| `active_days` | Number of unique trading days |

### 3. Target Variable
`comm_next` — the total commission each client generates in the **future** window. Clients with no future activity are assigned a target of 0 (rather than dropped), so the model also learns to correctly forecast low/zero LTV for clients likely to go dormant.

### 4. Modeling
A **Gradient Boosting Regressor** is trained to predict future commission from historical features:

```python
GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.05,
    max_depth=3, random_state=42
)
```

### 5. Evaluation
Model performance is assessed using:
- **MAE** (Mean Absolute Error) — average forecast error in commission currency units
- **R²** (coefficient of determination) — proportion of variance in future commission explained by historical behavior

### 6. Scoring & Tiering
The trained model forecasts LTV for **every** client, which are then split into four **quartile-based tiers**:

| Tier | Definition |
|---|---|
| 🥉 Bronze | Bottom 25% of forecasted LTV |
| 🥈 Silver | 25th–50th percentile |
| 🥇 Gold | 50th–75th percentile |
| 💎 Platinum | Top 25% of forecasted LTV |

Clients are sorted by forecasted LTV (highest first) and exported to `Life time value_tiers.csv` for use by RM teams.

## Results

### Model Performance (held-out test set)

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | 18,625.07 |
| R² Score | 0.382 |

The model explains roughly **38% of the variance** in future commission using only five historical behavioral features. An MAE of ~18.6K (in the dataset's commission currency units) indicates moderate forecasting precision — useful for relative ranking and tiering, though individual point forecasts should be treated as directional rather than exact.

> **Note on interpretation:** Unlike the companion churn model in this author's other repository, this R² is realistic rather than suspiciously perfect — future commission is inherently noisy and influenced by factors outside historical trading patterns (market conditions, client life events, etc.), so an R² of 0.38 is a believable, defensible result rather than a red flag.

### Client Tier Distribution

| Tier | Client Count | Avg. Forecasted LTV |
|---|---|---|
| 🥉 Bronze | 407 | $798.46 |
| 🥈 Silver | 407 | $3,273.09 |
| 🥇 Gold | 408 | $11,615.86 |
| 💎 Platinum | 406 | $77,097.64 |

### Key Takeaways

- **Value concentration is extreme:** the top 25% of clients (Platinum) are forecast to generate **~97x** the commission of the bottom 25% (Bronze) on average — a small segment likely drives a disproportionate share of future revenue.
- **Bronze clients, while individually low-value, are the largest segment by count** — worth monitoring in aggregate for growth potential and cost-to-serve efficiency, but not necessarily worth premium RM attention as individuals.
- **Gold clients represent the best growth target** — high current engagement with room to move into Platinum through deeper relationship management.

### Suggested Resource Allocation

| Tier | Recommended Strategy |
|---|---|
| 💎 Platinum | Dedicated senior RMs, white-glove service, proactive engagement and cross-sell |
| 🥇 Gold | Skilled RMs, tailored products, engagement strategies to drive promotion to Platinum |
| 🥈 Silver | Regular but standardized engagement, educational content, semi-automated touchpoints |
| 🥉 Bronze | Low-touch, automated engagement; monitor in aggregate for cost-efficient growth |

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab

### Installation

```bash
git clone https://github.com/Mdkamrulislam54/The-Client-Lifetime-Value-LTV-forecasting-model.git
cd The-Client-Lifetime-Value-LTV-forecasting-model
pip install -r requirements.txt
```

### Usage

1. Place `Financial Dataset.xlsx` in the project root (already included in this repo).
2. Open and run the notebook:

```bash
jupyter notebook The_Client_Lifetime_Value__LTV__forecasting_model.ipynb
```

3. Run all cells top to bottom. The notebook will:
   - Load and split data into historical/future windows
   - Engineer client-level features
   - Train and evaluate the regressor
   - Generate `Life time value_tiers.csv` with tier assignments

## Tech Stack

- **Python** — pandas, scikit-learn
- **Modeling** — Gradient Boosting Regressor
- **Environment** — Jupyter Notebook

## Future Improvements

- Add more historical features (recency, trade frequency, deposit/withdrawal net flow) to improve R²
- Test alternative time windows and rolling-window backtesting for robustness
- Add feature importance / SHAP analysis to explain what drives high-LTV forecasts
- Combine with the [Client Churn (Dormancy) Prediction](https://github.com/Mdkamrulislam54/Client-Churn-Dormancy-Prediction) model to jointly prioritize clients by **value at risk** (high LTV + high churn probability)
- Automate periodic re-scoring as new transaction data arrives

## License

This project is available under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Md Kamrul Islam**
GitHub: [@Mdkamrulislam54](https://github.com/Mdkamrulislam54)
