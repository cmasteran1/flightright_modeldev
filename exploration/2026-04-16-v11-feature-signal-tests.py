"""
v11 new-feature signal tests on WN data.

Three tasks:
(a) Late-aircraft rate: LateAircraftDelay > 0 (v10) vs >= 15 (v11)
(b) flightnum_od_otp_rate_last14d — binary OTP rate, marginal vs median
(c) airline_cancel_rate_anomaly_7d — z-score of 7d cancel rate vs 60d baseline

Data:
- Features + labels: features_dep_WN_v10_smoke_unbalanced.parquet (June 2025, 42k rows)
- Raw BTS history: 2025-04/05/06 (for computing features that need lookback)

All aggregates use the v10 leakage-prevention pattern: compute daily aggregate,
shift 1 day forward, merge on (key, FlightDate).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DATA_ROOT = Path("/Users/connermasteran/software/mpqc/flightrightdata")
REPO_ROOT = Path("/Users/connermasteran/software/mpqc/flightright_modeldev")

FEATURES_PATH = DATA_ROOT / "data/processed/features_dep_WN_v10_smoke_unbalanced.parquet"
BTS_FILES = [
    DATA_ROOT / "data/raw_bts/Year=2025/Month=04/flights.parquet",
    DATA_ROOT / "data/raw_bts/Year=2025/Month=05/flights.parquet",
    DATA_ROOT / "data/raw_bts/Year=2025/Month=06/flights.parquet",
]

BTS_COLS = [
    "FlightDate", "Operating_Airline ",
    "Flight_Number_Operating_Airline", "Origin", "Dest",
    "DepDelayMinutes", "LateAircraftDelay", "NASDelay",
    "Cancelled",
]

AIRLINE = "WN"

# ---------- load ----------

print("Loading features parquet...")
feats = pd.read_parquet(FEATURES_PATH)
feats["FlightDate"] = pd.to_datetime(feats["FlightDate"]).dt.normalize()
print(f"  Rows: {len(feats)}, date range: {feats['FlightDate'].min().date()} -> {feats['FlightDate'].max().date()}")

print("Loading raw BTS history...")
bts_list = []
for p in BTS_FILES:
    if p.exists():
        dfp = pd.read_parquet(p, columns=BTS_COLS)
        bts_list.append(dfp)
        print(f"  {p.parent.name}: {len(dfp)} rows")
bts = pd.concat(bts_list, ignore_index=True)
bts.columns = [c.strip() for c in bts.columns]
bts = bts.rename(columns={"Operating_Airline": "Carrier", "Flight_Number_Operating_Airline": "FlightNum"})
bts["FlightDate"] = pd.to_datetime(bts["FlightDate"]).dt.normalize()
for c in ["DepDelayMinutes", "LateAircraftDelay", "NASDelay", "Cancelled"]:
    bts[c] = pd.to_numeric(bts[c], errors="coerce")
print(f"  Total BTS rows: {len(bts)}")

# Build ge120 label from DepDelayMinutes in the features parquet
# (v11 needs y_dep_ge60 and y_dep_ge120; y_dep_ge60 already exists as y_dep_ge60 NO — it has ge45, not ge60)
# Actually we need to recreate via DepDelayMinutes
feats["y_dep_ge15"] = (feats["DepDelayMinutes"] >= 15).astype(int)
feats["y_dep_ge30"] = (feats["DepDelayMinutes"] >= 30).astype(int)
feats["y_dep_ge60"] = (feats["DepDelayMinutes"] >= 60).astype(int)
feats["y_dep_ge120"] = (feats["DepDelayMinutes"] >= 120).astype(int)
print()
print("Label base rates:")
for k in ["y_dep_ge15", "y_dep_ge30", "y_dep_ge60", "y_dep_ge120"]:
    print(f"  {k}: {feats[k].mean():.4f}")
print()

# ---------- Task (a): late aircraft rate at two thresholds ----------

print("=" * 70)
print("TASK (a): Late-aircraft rate — LateAircraftDelay > 0 vs >= 15")
print("=" * 70)


def compute_origin_la_rate(bts_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Daily origin-level late-aircraft rate, shifted 1 day forward."""
    h = bts_df.dropna(subset=["Origin", "FlightDate"]).copy()
    h["_has_la"] = (h["LateAircraftDelay"].fillna(0) >= threshold).astype(float)
    daily = h.groupby(["Origin", "FlightDate"], as_index=False)["_has_la"].mean()
    daily["FlightDate"] = daily["FlightDate"] + pd.Timedelta(days=1)
    return daily.rename(columns={"_has_la": "origin_la_rate"})


def compute_carrier_la_rate(bts_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    h = bts_df[bts_df["Carrier"] == AIRLINE].dropna(subset=["FlightDate"]).copy()
    h["_has_la"] = (h["LateAircraftDelay"].fillna(0) >= threshold).astype(float)
    daily = h.groupby(["FlightDate"], as_index=False)["_has_la"].mean()
    daily["FlightDate"] = daily["FlightDate"] + pd.Timedelta(days=1)
    return daily.rename(columns={"_has_la": "carrier_la_rate"})


# v11 variants (>=15)
origin_v11 = compute_origin_la_rate(bts, threshold=15).rename(columns={"origin_la_rate": "origin_la_rate_v11"})
carrier_v11 = compute_carrier_la_rate(bts, threshold=15).rename(columns={"carrier_la_rate": "carrier_la_rate_v11"})

feats_a = feats[["FlightDate", "Origin", "Reporting_Airline", "y_dep_ge15", "y_dep_ge60", "y_dep_ge120",
                 "origin_lateaircraft_rate_last1", "carrier_lateaircraft_rate_last1"]].copy()
feats_a = feats_a.merge(origin_v11, on=["Origin", "FlightDate"], how="left")
feats_a = feats_a.merge(carrier_v11, on=["FlightDate"], how="left")
feats_a = feats_a.dropna(subset=["origin_la_rate_v11", "carrier_la_rate_v11",
                                 "origin_lateaircraft_rate_last1", "carrier_lateaircraft_rate_last1"])
print(f"  Effective rows: {len(feats_a)}")

# Distribution shift
print()
print("Origin late-aircraft rate distribution:")
print(f"  v10 (>0):   mean={feats_a['origin_lateaircraft_rate_last1'].mean():.4f}  median={feats_a['origin_lateaircraft_rate_last1'].median():.4f}  p90={feats_a['origin_lateaircraft_rate_last1'].quantile(0.9):.4f}")
print(f"  v11 (>=15): mean={feats_a['origin_la_rate_v11'].mean():.4f}  median={feats_a['origin_la_rate_v11'].median():.4f}  p90={feats_a['origin_la_rate_v11'].quantile(0.9):.4f}")
print()
print("Carrier late-aircraft rate distribution:")
print(f"  v10 (>0):   mean={feats_a['carrier_lateaircraft_rate_last1'].mean():.4f}  median={feats_a['carrier_lateaircraft_rate_last1'].median():.4f}  p90={feats_a['carrier_lateaircraft_rate_last1'].quantile(0.9):.4f}")
print(f"  v11 (>=15): mean={feats_a['carrier_la_rate_v11'].mean():.4f}  median={feats_a['carrier_la_rate_v11'].median():.4f}  p90={feats_a['carrier_la_rate_v11'].quantile(0.9):.4f}")
print()

# Signal
print("Signal comparison (point-biserial + univariate AUC):")
results_a = []
for feat_name in ["origin_lateaircraft_rate_last1", "origin_la_rate_v11",
                  "carrier_lateaircraft_rate_last1", "carrier_la_rate_v11"]:
    for label in ["y_dep_ge15", "y_dep_ge60", "y_dep_ge120"]:
        x = feats_a[feat_name].values
        y = feats_a[label].values
        if y.std() == 0 or x.std() == 0:
            continue
        r, p = stats.pointbiserialr(y, x)
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            auc = np.nan
        results_a.append({"feature": feat_name, "label": label,
                          "pointbiserial_r": r, "auc": auc, "n": len(feats_a)})
        print(f"  {feat_name:40s} vs {label:15s} r={r:+.4f} auc={auc:.4f}")

df_results_a = pd.DataFrame(results_a)

# ---------- Task (b): flightnum_od_otp_rate_last14d ----------

print()
print("=" * 70)
print("TASK (b): flightnum_od_otp_rate_last14d")
print("=" * 70)


def compute_otp_rate(bts_df: pd.DataFrame, airline: str, window_days: int = 14, threshold: int = 15) -> pd.DataFrame:
    """
    For each (Carrier, FlightNum, Origin, Dest, FlightDate), compute the
    fraction of same-identity flights in the prior `window_days` with
    DepDelayMinutes <= threshold. Shifted 1 day forward to prevent leakage.
    """
    h = bts_df[bts_df["Carrier"] == airline].copy()
    h = h.dropna(subset=["FlightDate", "FlightNum", "Origin", "Dest", "DepDelayMinutes"])
    # Exclude cancelled flights
    h = h[h["Cancelled"].fillna(0) == 0]
    h["_on_time"] = (h["DepDelayMinutes"] <= threshold).astype(float)

    # Group and daily mean
    daily = (h.groupby(["Carrier", "FlightNum", "Origin", "Dest", "FlightDate"], as_index=False)
             .agg(on_time_mean=("_on_time", "mean"), n=("_on_time", "size"))
             .sort_values(["Carrier", "FlightNum", "Origin", "Dest", "FlightDate"]))

    # Rolling 14-day mean (weighted by count)
    # Simple: rolling mean over daily means. Good enough for low-cardinality flight identities.
    def _roll(g):
        g = g.sort_values("FlightDate").copy()
        g["otp_rate_14d"] = g["on_time_mean"].shift(1).rolling(window_days, min_periods=1).mean()
        return g

    daily = daily.groupby(["Carrier", "FlightNum", "Origin", "Dest"], group_keys=False).apply(_roll)
    return daily[["Carrier", "FlightNum", "Origin", "Dest", "FlightDate", "otp_rate_14d"]]


otp = compute_otp_rate(bts, AIRLINE, 14, 15)
print(f"  Computed OTP rate for {len(otp)} (flightnum, OD, date) groups")

# Merge onto features. Features parquet uses Reporting_Airline and Flight_Number_Reporting_Airline
feats_b_cols = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
                "y_dep_ge15", "y_dep_ge60", "y_dep_ge120",
                "flightnum_od_depdelay_median_last14", "flightnum_od_depdelay_mean_last14"]
feats_b = feats[feats_b_cols].rename(columns={"Reporting_Airline": "Carrier", "Flight_Number_Reporting_Airline": "FlightNum"}).copy()
feats_b = feats_b.merge(otp, on=["Carrier", "FlightNum", "Origin", "Dest", "FlightDate"], how="left")

n_with_otp = feats_b["otp_rate_14d"].notna().sum()
print(f"  Rows with OTP rate: {n_with_otp} / {len(feats_b)} ({100*n_with_otp/len(feats_b):.1f}%)")

# Univariate signal
print()
print("Univariate signal (OTP rate has inverse relationship with delay; expect negative r):")
b_feats = feats_b.dropna(subset=["otp_rate_14d", "flightnum_od_depdelay_median_last14",
                                 "flightnum_od_depdelay_mean_last14"])
results_b = []
for label in ["y_dep_ge15", "y_dep_ge60", "y_dep_ge120"]:
    y = b_feats[label].values
    if y.std() == 0:
        continue
    for feat_name in ["otp_rate_14d", "flightnum_od_depdelay_median_last14", "flightnum_od_depdelay_mean_last14"]:
        x = b_feats[feat_name].values
        r, _ = stats.spearmanr(x, y)
        try:
            # For OTP (inverse relationship), AUC with -x
            auc = roc_auc_score(y, -x if feat_name == "otp_rate_14d" else x)
        except ValueError:
            auc = np.nan
        print(f"  {feat_name:45s} vs {label:15s} rho={r:+.4f} auc={auc:.4f}")
        results_b.append({"feature": feat_name, "label": label,
                          "spearman_r": r, "auc": auc, "n": len(b_feats)})

# Partial correlation: OTP rate controlling for median (linear regression residuals)
print()
print("Partial Spearman of otp_rate_14d controlling for flightnum_od_depdelay_median_last14:")
from sklearn.linear_model import LinearRegression
for label in ["y_dep_ge15", "y_dep_ge60", "y_dep_ge120"]:
    if b_feats[label].std() == 0:
        continue
    lr_x = LinearRegression().fit(b_feats[["flightnum_od_depdelay_median_last14"]], b_feats["otp_rate_14d"])
    resid_x = b_feats["otp_rate_14d"] - lr_x.predict(b_feats[["flightnum_od_depdelay_median_last14"]])
    lr_y = LinearRegression().fit(b_feats[["flightnum_od_depdelay_median_last14"]], b_feats[label])
    resid_y = b_feats[label] - lr_y.predict(b_feats[["flightnum_od_depdelay_median_last14"]])
    r_partial, _ = stats.spearmanr(resid_x, resid_y)
    print(f"  vs {label:15s} partial_rho={r_partial:+.4f}")

# Marginal log-loss improvement via simple logistic model
print()
print("Univariate logistic log-loss (lower is better):")
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
b_full = feats_b.dropna(subset=["otp_rate_14d", "flightnum_od_depdelay_median_last14"])
for label in ["y_dep_ge15", "y_dep_ge60", "y_dep_ge120"]:
    y = b_full[label].values
    if y.std() == 0:
        continue
    # Base model: median only
    X_base = b_full[["flightnum_od_depdelay_median_last14"]].values
    X_with = b_full[["flightnum_od_depdelay_median_last14", "otp_rate_14d"]].values
    Xa_tr, Xa_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.3, random_state=0)
    Xb_tr, Xb_te, _, _ = train_test_split(X_with, y, test_size=0.3, random_state=0)
    ll_base = log_loss(y_te, LogisticRegression(max_iter=200).fit(Xa_tr, y_tr).predict_proba(Xa_te))
    ll_with = log_loss(y_te, LogisticRegression(max_iter=200).fit(Xb_tr, y_tr).predict_proba(Xb_te))
    delta = ll_base - ll_with
    pct = 100 * delta / ll_base
    print(f"  {label:15s} base={ll_base:.4f} with_otp={ll_with:.4f} delta={delta:+.5f} ({pct:+.2f}%)")

# ---------- Task (c): airline_cancel_rate_anomaly_7d ----------

print()
print("=" * 70)
print("TASK (c): airline_cancel_rate_anomaly_7d")
print("=" * 70)


def compute_cancel_anomaly(bts_df: pd.DataFrame, airline: str) -> pd.DataFrame:
    """Z-score of 7d cancel rate vs 60d baseline."""
    h = bts_df[bts_df["Carrier"] == airline].dropna(subset=["FlightDate"]).copy()
    h["_cancelled"] = h["Cancelled"].fillna(0).astype(float)

    daily = h.groupby("FlightDate", as_index=False)["_cancelled"].mean().sort_values("FlightDate")
    daily["cancel_rate_7d"] = daily["_cancelled"].shift(1).rolling(7, min_periods=1).mean()
    daily["cancel_rate_60d"] = daily["_cancelled"].shift(1).rolling(60, min_periods=7).mean()
    daily["cancel_rate_60d_std"] = daily["_cancelled"].shift(1).rolling(60, min_periods=7).std()
    daily["cancel_rate_anomaly_7d"] = (daily["cancel_rate_7d"] - daily["cancel_rate_60d"]) / daily["cancel_rate_60d_std"].replace(0, np.nan)
    daily["FlightDate"] = daily["FlightDate"] + pd.Timedelta(days=1)
    return daily[["FlightDate", "cancel_rate_anomaly_7d", "cancel_rate_7d", "cancel_rate_60d"]]


cancel_anom = compute_cancel_anomaly(bts, AIRLINE)
n_finite = cancel_anom["cancel_rate_anomaly_7d"].notna().sum()
print(f"  Computed cancel anomaly for {n_finite} daily rows")
print(f"  Anomaly distribution: mean={cancel_anom['cancel_rate_anomaly_7d'].mean():.3f} std={cancel_anom['cancel_rate_anomaly_7d'].std():.3f}")

feats_c = feats[["FlightDate", "y_dep_ge15", "y_dep_ge60", "y_dep_ge120"]].copy()
feats_c = feats_c.merge(cancel_anom, on="FlightDate", how="left")
c_rows = feats_c.dropna(subset=["cancel_rate_anomaly_7d"])
print(f"  Features rows with anomaly: {len(c_rows)} / {len(feats_c)}")

print()
print("Signal:")
results_c = []
for label in ["y_dep_ge15", "y_dep_ge60", "y_dep_ge120"]:
    y = c_rows[label].values
    if y.std() == 0:
        continue
    x = c_rows["cancel_rate_anomaly_7d"].values
    rho, _ = stats.spearmanr(x, y)
    r, _ = stats.pointbiserialr(y, x)
    try:
        auc = roc_auc_score(y, x)
    except ValueError:
        auc = np.nan
    print(f"  cancel_rate_anomaly_7d vs {label:15s} rho={rho:+.4f} r={r:+.4f} auc={auc:.4f}")
    results_c.append({"feature": "cancel_rate_anomaly_7d", "label": label,
                      "spearman_r": rho, "pointbiserial_r": r, "auc": auc, "n": len(c_rows)})

# Save all results to CSVs for the report
out_dir = REPO_ROOT / "exploration/reports"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results_a).to_csv(out_dir / "2026-04-16-task-a-la-threshold.csv", index=False)
pd.DataFrame(results_b).to_csv(out_dir / "2026-04-16-task-b-otp-rate.csv", index=False)
pd.DataFrame(results_c).to_csv(out_dir / "2026-04-16-task-c-cancel-anomaly.csv", index=False)

print()
print("Done. Raw results saved to exploration/reports/2026-04-16-task-*.csv")
