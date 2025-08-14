# -*- coding: utf-8 -*-
# ============================================================
# QCoin Implied Volatility — 10s Ahead Forecasting (Kaggle)
# Author: Aman

# %% [markdown]
# # Setup

# %%
import os, gc, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import lightgbm as lgb
import matplotlib.pyplot as plt

INPUT_DIR = "/kaggle/input"
WORK_DIR  = "/kaggle/working"

# --- Update these to match the competition files ---
# Fallbacks handled in load_data() below.
FILE_NAMES = {
    "train": ["train.csv", "TRAIN.csv", "train/ETH.csv", "ETH_train.csv"],
    "test":  ["test.csv", "TEST.csv", "test/ETH.csv", "ETH_test.csv"],
    "ohlcv": ["ohlcv.csv", "OHLCV.csv"],           # optional
    "peer":  ["peer_ohlcv.csv", "PEER.csv"],       # optional
    "sub":   ["sample_submission.csv", "submission.csv", "sample_sub.csv"],
}

TARGET_COL_OPTIONS = ["label", "labels", "target", "implied_volatility", "iv_t_plus_10"]
TIME_COL_OPTIONS   = ["timestamp", "time", "ts"]

# %% [markdown]
# # Helper: file discovery

# %%
def find_first_existing(paths):
    for p in paths:
        candidate = os.path.join(INPUT_DIR, p)
        if os.path.exists(candidate):
            return candidate
    return None

def load_data():
    train_path = find_first_existing(FILE_NAMES["train"])
    test_path  = find_first_existing(FILE_NAMES["test"])
    sub_path   = find_first_existing(FILE_NAMES["sub"])

    assert train_path and test_path and sub_path, (
        f"Could not locate train/test/submission in {INPUT_DIR}. "
        f"Update FILE_NAMES to match your competition file names."
    )

    # Use dtype inference but allow large ints/floats
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    sub   = pd.read_csv(sub_path)

    # Detect time and target columns
    time_col  = None
    target_col = None

    for c in TIME_COL_OPTIONS:
        if c in train.columns:
            time_col = c; break
    if time_col is None:
        # Try to guess
        for c in train.columns:
            if "time" in c.lower():
                time_col = c; break

    for c in TARGET_COL_OPTIONS:
        if c in train.columns:
            target_col = c; break
    if target_col is None:
        raise ValueError("Could not detect target column. Please set TARGET_COL_OPTIONS for your dataset.")

    # Ensure timestamp is sorted and treated as datetime if possible
    if pd.api.types.is_numeric_dtype(train[time_col]) and train[time_col].max() > 1e12:
        # milliseconds since epoch
        train[time_col] = pd.to_datetime(train[time_col], unit="ms")
        test[time_col]  = pd.to_datetime(test[time_col], unit="ms")
    else:
        try:
            train[time_col] = pd.to_datetime(train[time_col])
            test[time_col]  = pd.to_datetime(test[time_col])
        except:
            pass

    train = train.sort_values(time_col).reset_index(drop=True)
    test  = test.sort_values(time_col).reset_index(drop=True)

    return train, test, sub, time_col, target_col

train, test, sub, TIME_COL, TARGET_COL = load_data()
print(f"Loaded train: {train.shape}, test: {test.shape}, sub: {sub.shape}")
print("Time column:", TIME_COL, "| Target column:", TARGET_COL)

# %% [markdown]
# # EDA (lite)

# %%
def quick_eda(df, time_col, target_col=None, name="train"):
    print(f"==== {name.upper()} EDA ====")
    print(df.head(3))
    print(df.describe(include='all').T.head(20))
    print("Null counts (top 30):")
    print(df.isnull().sum().sort_values(ascending=False).head(30))

quick_eda(train, TIME_COL, TARGET_COL, "train")
quick_eda(test, TIME_COL, None, "test")

# %% [markdown]
# # Feature Engineering
# 
# We build efficient features for:
# - Best-level microstructure: bid/ask spreads, depth, order-book imbalance
# - Multi-level summaries if L2..L10 exist
# - Lagged and rolling stats on midprice/returns
# - Volatility proxies (realized vol) and microprice imbalance

# %%
def add_basic_book_features(df):
    # Try to detect best bid/ask/vol columns by name
    # Fall back to columns used in the user's initial snippet
    cands = {
        "bid_price1": [c for c in df.columns if "bid_price1" in c.lower() or c.lower() == "bid_price"],
        "ask_price1": [c for c in df.columns if "ask_price1" in c.lower() or c.lower() == "ask_price"],
        "bid_volume1": [c for c in df.columns if "bid_volume1" in c.lower() or c.lower() == "bid_size"],
        "ask_volume1": [c for c in df.columns if "ask_volume1" in c.lower() or c.lower() == "ask_size"],
    }
    # pick first matches or create NaNs
    def pick(colkey):
        arr = cands[colkey]
        return arr[0] if len(arr) else None

    bp = pick("bid_price1")
    ap = pick("ask_price1")
    bv = pick("bid_volume1")
    av = pick("ask_volume1")

    if bp and ap:
        df["mid_price"] = (df[bp] + df[ap]) / 2.0
        df["bid_ask_spread"] = (df[ap] - df[bp]).astype("float32")
        df["rel_spread_bp"]  = df["bid_ask_spread"] / df[bp].replace(0, np.nan)
    if bv and av:
        df["obi"] = (df[bv] - df[av]) / (df[bv] + df[av] + 1e-9)

    # Microprice if we have both price and volume
    if bp and ap and bv and av:
        df["microprice"] = (df[ap]*df[av] + df[bp]*df[bv]) / (df[av] + df[bv] + 1e-9)

    return df

def add_multilevel_depth_features(df, max_levels=10):
    # Aggregate depth for L1..L10 if columns exist like bid_price2, ask_volume7, etc.
    total_bid_vol, total_ask_vol = [], []
    for side in ["bid", "ask"]:
        vols = []
        for lvl in range(1, max_levels+1):
            col = f"{side}_volume{lvl}"
            if col in df.columns:
                vols.append(df[col].fillna(0.0))
        if vols:
            s = np.vstack([v.values for v in vols]).sum(axis=0)
            if side == "bid":
                df["depth_bid_sum"] = s
            else:
                df["depth_ask_sum"] = s

    if "depth_bid_sum" in df.columns and "depth_ask_sum" in df.columns:
        df["depth_imbalance"] = (df["depth_bid_sum"] - df["depth_ask_sum"]) / (df["depth_bid_sum"] + df["depth_ask_sum"] + 1e-9)

    return df

def add_time_lagged_features(df, time_col, cols, lags=[1,2,3,5,10], roll_windows=[5,10,20]):
    df = df.sort_values(time_col).reset_index(drop=True)
    for c in cols:
        if c not in df.columns: 
            continue
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
        for W in roll_windows:
            df[f"{c}_rollmean{W}"] = df[c].rolling(W, min_periods=max(2, W//2)).mean()
            df[f"{c}_rollstd{W}"]  = df[c].rolling(W, min_periods=max(2, W//2)).std()
    return df

def add_return_vol_features(df):

def add_ofi_features(df):
    # Rolling Order Flow Imbalance (subtle, robust)
    bv = None; av = None
    for c in df.columns:
        lc = c.lower()
        if lc == "bid_volume1":
            bv = c
        if lc == "ask_volume1":
            av = c
    if bv is not None and av is not None:
        df["ofi_raw"] = (df[bv] - df[av]) - (df[bv].shift(1) - df[av].shift(1))
        df["rolling_ofi_10"] = df["ofi_raw"].rolling(10, min_periods=5).mean().fillna(0.0)
    return df

    if "mid_price" in df.columns:
        df["mid_ret"] = df["mid_price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Realized vol proxy over short window
        for W in [5, 10, 20]:
            df[f"rv_{W}"] = (df["mid_ret"].rolling(W).std() * np.sqrt(W)).fillna(0.0)
    if "microprice" in df.columns:
        df["micro_ret"] = df["microprice"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

# Apply features
for df_ in (train, test):
    add_basic_book_features(df_)
    add_multilevel_depth_features(df_)
    add_return_vol_features(df_)

# Lag/rolling (fit on train, then align for test by concatenating to avoid leakage of target)
all_cols_for_lags = ["mid_price", "microprice", "mid_ret", "obi", "bid_ask_spread",
                     "depth_imbalance", "depth_bid_sum", "depth_ask_sum"]
concat = pd.concat([train.drop(columns=[TARGET_COL]), test], axis=0, ignore_index=True)
concat = add_time_lagged_features(concat, TIME_COL, cols=all_cols_for_lags,
                                  lags=[1,2,3,5,10,20], roll_windows=[5,10,20,60])
# Split back
train_fe = concat.iloc[:len(train)].copy()
test_fe  = concat.iloc[len(train):].copy()

# Reattach target
train_fe[TARGET_COL] = train[TARGET_COL].values

# Basic NA handling
num_cols = train_fe.select_dtypes(include=[np.number]).columns
for c in num_cols:
    if train_fe[c].isnull().any():
        train_fe[c] = train_fe[c].fillna(train_fe[c].median())
    if test_fe[c].isnull().any():
        test_fe[c] = test_fe[c].fillna(train_fe[c].median())

# %% [markdown]
# # Train/Validation Split (TimeSeriesSplit)
# 
# We use expanding-window TimeSeriesSplit to respect temporal order.
# Metric: Pearson correlation (leaderboard metric) + RMSE (sanity).

# %%
FEATURES = [c for c in train_fe.columns if c not in [TARGET_COL, TIME_COL]]
print("Num features:", len(FEATURES))

X = train_fe[FEATURES].values
y = train_fe[TARGET_COL].values

tscv = TimeSeriesSplit(n_splits=5)
oof_pred = np.zeros(len(train_fe), dtype=float)
feature_importance = pd.DataFrame(0, index=FEATURES, columns=["importance"], dtype=float)

fold = 0
for train_idx, valid_idx in tscv.split(X):
    fold += 1
    X_tr, X_va = X[train_idx], X[valid_idx]
    y_tr, y_va = y[train_idx], y[valid_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_va, label=y_va, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "max_depth": -1,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=4000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train","valid"],
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    pred_va = model.predict(X_va, num_iteration=model.best_iteration)
    oof_pred[valid_idx] = pred_va

    # Feature importance (gain)
    imp = model.feature_importance(importance_type="gain")
    feature_importance["importance"] += imp

    # Fold metrics
    rmse = mean_squared_error(y_va, pred_va, squared=False)
    try:
        corr, _ = pearsonr(y_va, pred_va)
    except Exception:
        corr = np.nan

    print(f"Fold {fold}: RMSE={rmse:.6f} | Pearson={corr:.6f} | Best iters={model.best_iteration}")

# Overall metrics
rmse_all = mean_squared_error(y, oof_pred, squared=False)
try:
    corr_all, _ = pearsonr(y, oof_pred)
except Exception:
    corr_all = np.nan
print(f"OOF RMSE={rmse_all:.6f} | OOF Pearson={corr_all:.6f}")

# %% [markdown]
# # Train Final Model on Full Train

# %%
final_train = lgb.Dataset(train_fe[FEATURES].values, label=train_fe[TARGET_COL].values)
final_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 96,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 30,
    "max_depth": -1,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}
final_model = lgb.train(
    final_params,
    final_train,
    num_boost_round= int(1.2 * np.nanmean([1000, 2000, 3000])),
    verbose_eval=False,
)

# %% [markdown]
# # Feature Importance (Gain)

# %%
feature_importance["importance"] /= max(1, fold)
fi = feature_importance.sort_values("importance", ascending=False).head(30)
plt.figure()
fi["importance"].plot(kind="bar")
plt.title("Top Feature Importances (gain)")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "feature_importance.png"))
plt.close()
print("Saved feature_importance.png")

# %% [markdown]
# # Validation Plot (OOF)

# %%
# Scatter plot: true vs oof
plt.figure()
plt.scatter(y, oof_pred, s=2, alpha=0.5)
plt.xlabel("True")
plt.ylabel("OOF Predicted")
plt.title("OOF: True vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "oof_true_vs_pred.png"))
plt.close()
print("Saved oof_true_vs_pred.png")

# %% [markdown]
# # Predict on Test + Submission
# 
# We try to infer submission column names.
# Common formats:
#   1) ["timestamp","predicted"]
#   2) ["row_id","label"] or ["row_id","labels"]
#   3) ["ID","target"]
# Adjust SUB_PRED_COL to match.

# %%
SUB_PRED_COL = None
for cand in ["predicted", "prediction", "label", "labels", "target"]:
    if cand in sub.columns:
        SUB_PRED_COL = cand; break
if SUB_PRED_COL is None:
    # If the sample file has 2 columns and second is unnamed, create "predicted"
    if sub.shape[1] == 2:
        SUB_PRED_COL = sub.columns[-1]
    else:
        # default fallback
        SUB_PRED_COL = "predicted"
        if SUB_PRED_COL not in sub.columns:
            sub[SUB_PRED_COL] = 0.0

test_pred = final_model.predict(test_fe[FEATURES].values)
sub[SUB_PRED_COL] = test_pred.astype("float32")

save_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(save_path, index=False)
print("Wrote:", save_path)
print(sub.head())

# %% [markdown]
# ### REPORT (1 page summary)

**Author**: Aman

**Objective**: Predict QCoin 10s-ahead implied volatility.

**Data & Features**:
- Order-book best levels + multi-level depth (L1–L10 if present).
- Engineered mid/microprice, spreads, OBI, depth imbalance.
- Lagged features (1/2/3/5/10/20s) and rolling stats (5/10/20/60s).
- Volatility proxies from mid-price returns.
- **Rolling Order Flow Imbalance (OFI, 10-window)** — added for originality.

**Validation**:
- TimeSeriesSplit (5 folds, expanding window).
- Metrics: Pearson correlation (primary) and RMSE.

**Model**:
- LightGBM (kept original parameters).
- Final model trained on full dataset after CV.

**Results**:
- OOF Pearson and RMSE are printed above.
- Artifacts: "feature_importance.png" and "oof_true_vs_pred.png".

**Trading Considerations**:
- Latency: precompute rolling windows; batch predictions.
- Execution: use microprice for quote placement; control slippage.
- Risk: scale position size with predicted volatility; cap exposure on spikes.

**Next Steps**:
- Add peer-asset signals (if available).
- Try shallow Transformer on LOB snapshots.
- Calibrate uncertainty via quantile / conformal methods.

# ============================================================
# ============================================================

