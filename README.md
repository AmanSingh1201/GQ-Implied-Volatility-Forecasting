# QCoin Implied Volatility — 10s Ahead Forecasting (Kaggle)

**Author:** Aman  
**Date:** 2025-08-14

## 📌 Overview
This project builds a **time-series forecasting pipeline** to predict QCoin's implied volatility (IV) **10 seconds ahead** from high-frequency **order book** and OHLCV style data. It is fully compatible with the Kaggle environment and adheres to the competition's evaluation metric (**Pearson Correlation**). The notebook generates a valid `submission.csv` and well-labeled diagnostic plots.

## 🛠 Skills & Tools (ATS Keywords)
LightGBM, Time Series Forecasting, Feature Engineering, Order Book Analysis, Market Microstructure, Volatility Modeling, Pearson Correlation, Cross-Validation, Pandas, NumPy, Scikit-learn, Matplotlib, Kaggle Notebooks

## 🔍 My Role & Contribution
- Implemented end-to-end pipeline—from data loading to leaderboard submission.
- Engineered microstructure features: mid/microprice, bid–ask spread, **order book imbalance (OBI)**, depth sums/imbalance.
- Added **Rolling Order Flow Imbalance (OFI)** (10-second window) as a unique, subtle feature.
- Used **expanding-window TimeSeriesSplit** to avoid look-ahead bias.
- Produced a clean `submission.csv` (`timestamp,predicted`) and documented assumptions.

## 📂 Files
- `qcoin_iv_kaggle_notebook.py` — Kaggle-ready pipeline (auto-split into cells).
- `submission.csv` — Predictions in competition format.
- `qcoin_iv_report.pdf` — 1-page summary report (objective → features → validation → results → trading use).


## ✅ How to Run on Kaggle
1. Kaggle → **Code → New Notebook → Upload** → select `qcoin_iv_kaggle_notebook.py`.
2. Right sidebar → **Add Data** → attach the competition dataset.
3. **Run all**. Outputs in `/kaggle/working/`:
   - `submission.csv`
   - `feature_importance.png`
   - `oof_true_vs_pred.png`
   - Printed OOF **Pearson** and **RMSE**
4. **Submit to Competition** using the generated `submission.csv`.
5. Share the **private notebook** with GoQuant usernames (view access).

## 📈 Validation & Metric
- 5-fold **TimeSeriesSplit** (expanding window) to simulate forward-in-time training.
- Primary metric: **Pearson Correlation**; secondary: RMSE for calibration.

## 💼 Business Impact
- Improves **options pricing** and **risk-managed execution**.
- Supports **market-making** and **hedging** via short-horizon IV signals.
- Enables **position sizing** and **risk targeting** using predicted volatility.

## 🧪 Assumptions & Libraries
- Timestamp column present/detectable as `timestamp`/`time`/`ts`.
- Target present as one of `label`, `labels`, `target`, `implied_volatility`, `iv_t_plus_10`.
- Libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `matplotlib` (Kaggle defaults).

## 🚀 Next Steps
- Add peer-asset (BTC/ETH) returns/vol if provided.
- Blend tree model with a lightweight Transformer over LOB snapshots.
- Uncertainty bands via quantile regression / conformal methods.

**Contact**: Aman — [amankumarara990@gmail.com] | LinkedIn: <https://www.linkedin.com/in/aman-profile/> 
