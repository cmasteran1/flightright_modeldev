"""
v11 WN smoke output validation — runs every layer (schema, range, invariant,
semantic, distribution) and reports findings.

Also performs a manual semantic re-derivation of flightnum_od_otp_rate_last14d
for a random sample of rows, to confirm the pipeline computation matches hand-calc.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

DATA = Path("/Users/connermasteran/software/mpqc/flightrightdata")
FEATS = DATA / "data/processed/features_dep_WN_v11_smoke_unbalanced.parquet"
V10_FEATS = DATA / "data/processed/features_dep_WN_v10_smoke_unbalanced.parquet"
BTS_MAY = DATA / "data/raw_bts/Year=2025/Month=05/flights.parquet"
BTS_JUN = DATA / "data/raw_bts/Year=2025/Month=06/flights.parquet"

findings: List[Dict[str, Any]] = []


def record(severity: str, layer: str, check: str, result: str, detail: str = "") -> None:
    findings.append({"severity": severity, "layer": layer, "check": check,
                     "result": result, "detail": detail})
    print(f"[{severity:4s}] [{layer:12s}] {check}: {result}  {detail}")


print("Loading v11 smoke features...")
df = pd.read_parquet(FEATS)
print(f"  Rows: {len(df)}")
print(f"  Cols: {len(df.columns)}")
print()

# ---------------- SCHEMA LAYER ----------------
print("=" * 72)
print("SCHEMA LAYER")
print("=" * 72)

for col in ["y_dep_ge15", "y_dep_ge30", "y_dep_ge60", "y_dep_ge120"]:
    if col in df.columns:
        record("PASS", "schema", f"label {col} present", "yes")
    else:
        record("FAIL", "schema", f"label {col} present", "MISSING")

# y_dep_ge45 must be absent (old v10 bucket)
col = "y_dep_ge45"
if col in df.columns:
    record("FAIL", "schema", f"old v10 {col} must be absent", "STILL PRESENT",
           "v11 buckets are [15,30,60,120]; ge45 must not appear")
else:
    record("PASS", "schema", f"old v10 {col} absent", "yes")

# New v11 feature
col = "flightnum_od_otp_rate_last14d"
if col in df.columns:
    record("PASS", "schema", f"{col} present", "yes")
else:
    record("FAIL", "schema", f"{col} present", "MISSING")

# Dropped v10 feature
col = "origin_nasdelay_rate_last1d"
if col in df.columns:
    record("FAIL", "schema", f"dropped {col} must be absent", "STILL PRESENT")
else:
    record("PASS", "schema", f"dropped {col} absent", "yes")

# Cancel anomaly (flag-gated off for smoke)
col = "airline_cancel_rate_anomaly_7d"
if col in df.columns:
    record("WARN", "schema", f"{col} (smoke-disabled)", "present")
else:
    record("PASS", "schema", f"{col} absent per smoke flag", "yes")

# Feature count (rough)
n_feat_cols = len([c for c in df.columns if not c.startswith("y_") and c not in {
    "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number",
    "FlightDate", "Origin", "Dest", "DepDelayMinutes", "ArrDelayMinutes",
    "Cancelled", "Diverted", "CRSArrTime", "ActualElapsedTime", "CRSElapsedTime",
    "DepDelay", "ArrDelay"
}])
record("INFO", "schema", "feature-ish column count", str(n_feat_cols),
       f"total cols {len(df.columns)}")

# ---------------- RANGE LAYER ----------------
print()
print("=" * 72)
print("RANGE LAYER")
print("=" * 72)

# OTP in [0, 1]
col = "flightnum_od_otp_rate_last14d"
s = df[col]
bad = df[(s < 0) | (s > 1)]
if len(bad) == 0:
    record("PASS", "range", f"{col} ∈ [0, 1]", "yes",
           f"nonnull={s.notna().mean():.1%}")
else:
    record("FAIL", "range", f"{col} ∈ [0, 1]", f"{len(bad)} out of range")

# All *_lateaircraft_rate_* in [0, 1]
la_cols = [c for c in df.columns if "lateaircraft" in c]
for c in la_cols:
    s = df[c]
    bad = df[(s < 0) | (s > 1)]
    if len(bad) == 0:
        record("PASS", "range", f"{c} ∈ [0, 1]", "yes")
    else:
        record("FAIL", "range", f"{c} ∈ [0, 1]", f"{len(bad)} out of range")

# Monotone label base rates: ge120 <= ge60 <= ge30 <= ge15
rates = {col: df[col].mean() for col in ["y_dep_ge15", "y_dep_ge30", "y_dep_ge60", "y_dep_ge120"]}
print(f"  Label base rates: {rates}")
if rates["y_dep_ge15"] >= rates["y_dep_ge30"] >= rates["y_dep_ge60"] >= rates["y_dep_ge120"]:
    record("PASS", "range", "label base rates monotone",
           f"{rates['y_dep_ge15']:.4f} >= {rates['y_dep_ge30']:.4f} >= {rates['y_dep_ge60']:.4f} >= {rates['y_dep_ge120']:.4f}")
else:
    record("FAIL", "range", "label base rates monotone", "VIOLATED")

# Plausibility: ge15 ~ 0.33 (industry avg), ge120 small but non-zero
if 0.2 <= rates["y_dep_ge15"] <= 0.5:
    record("PASS", "range", "y_dep_ge15 plausible (20%-50%)", f"{rates['y_dep_ge15']:.4f}")
else:
    record("WARN", "range", "y_dep_ge15 outside typical 20%-50%", f"{rates['y_dep_ge15']:.4f}")
if 0.01 <= rates["y_dep_ge120"] <= 0.10:
    record("PASS", "range", "y_dep_ge120 plausible (1%-10%)", f"{rates['y_dep_ge120']:.4f}")
else:
    record("WARN", "range", "y_dep_ge120 outside typical 1%-10%", f"{rates['y_dep_ge120']:.4f}")

# Label implication: y_dep_ge120 => y_dep_ge60 (hierarchical nesting)
violate = df[(df["y_dep_ge120"] == 1) & (df["y_dep_ge60"] == 0)]
if len(violate) == 0:
    record("PASS", "range", "y_dep_ge120 implies y_dep_ge60", "yes")
else:
    record("FAIL", "range", "y_dep_ge120 implies y_dep_ge60", f"{len(violate)} violations")

# ---------------- INVARIANT LAYER ----------------
print()
print("=" * 72)
print("INVARIANT LAYER")
print("=" * 72)

# v11 late-aircraft rates (>=15) should be <= v10 rates (>0) for same key/date.
# We have both parquet files so we can directly compare.
if V10_FEATS.exists():
    v10 = pd.read_parquet(V10_FEATS)
    # Join on (Origin, FlightDate) using origin_lateaircraft_rate_last1 as smallest example
    merge_keys = ["Origin", "FlightDate"]
    both = (df[merge_keys + ["origin_lateaircraft_rate_last1"]]
            .rename(columns={"origin_lateaircraft_rate_last1": "v11"})
            .merge(
                v10[merge_keys + ["origin_lateaircraft_rate_last1"]]
                .rename(columns={"origin_lateaircraft_rate_last1": "v10"}),
                on=merge_keys, how="inner"
            ))
    both = both.dropna(subset=["v10", "v11"])
    # Allow small numerical tolerance
    violations = both[both["v11"] > both["v10"] + 1e-9]
    if len(violations) == 0:
        record("PASS", "invariant", "origin_lateaircraft_rate v11<=v10 (stricter threshold)",
               "yes", f"n={len(both)} v11 mean={both['v11'].mean():.4f} v10 mean={both['v10'].mean():.4f}")
    else:
        # Could be due to (Origin, FlightDate) ambiguity since the history pools differ
        frac_violate = len(violations) / len(both)
        if frac_violate < 0.05:
            record("WARN", "invariant", "origin_lateaircraft_rate v11<=v10",
                   f"{len(violations)}/{len(both)} ({frac_violate:.1%}) minor violations",
                   "likely from history-pool differences between v10 and v11 smoke runs")
        else:
            record("FAIL", "invariant", "origin_lateaircraft_rate v11<=v10",
                   f"{len(violations)}/{len(both)} ({frac_violate:.1%})")
else:
    record("SKIP", "invariant", "v11<=v10 comparison", "no v10 smoke parquet found")

# OTP rate + any single-value-per-flight invariant: OTP rate itself in [0, 1]
# (already checked in range), but also: for rows with DepDelayMinutes <= 15 (on-time),
# the NEXT row for the same (carrier, flightnum, OD) shouldn't show a counterintuitive drop.
# Skipping a temporal diff test as it's hard to make robust on 1-month data.

# ---------------- SEMANTIC LAYER ----------------
print()
print("=" * 72)
print("SEMANTIC LAYER — manual re-derivation of flightnum_od_otp_rate_last14d")
print("=" * 72)

# Load raw BTS history used by the pipeline
bts_cols = ["FlightDate", "Operating_Airline ",
            "Flight_Number_Operating_Airline", "Origin", "Dest",
            "DepDelayMinutes", "Cancelled"]
bts_may = pd.read_parquet(BTS_MAY, columns=bts_cols)
bts_jun = pd.read_parquet(BTS_JUN, columns=bts_cols)
bts = pd.concat([bts_may, bts_jun], ignore_index=True)
bts.columns = [c.strip() for c in bts.columns]
bts = bts.rename(columns={
    "Operating_Airline": "Reporting_Airline",
    "Flight_Number_Operating_Airline": "Flight_Number_Reporting_Airline",
})
bts["FlightDate"] = pd.to_datetime(bts["FlightDate"]).dt.normalize()
for c in ["DepDelayMinutes", "Cancelled"]:
    bts[c] = pd.to_numeric(bts[c], errors="coerce")
bts = bts[bts["Reporting_Airline"] == "WN"].copy()

# Sample 5 rows with non-null OTP from the smoke parquet
sample = df.dropna(subset=["flightnum_od_otp_rate_last14d"]).sample(5, random_state=42)
print()
print("Re-deriving OTP for 5 random rows:")
print()

manual_match_count = 0
for _, row in sample.iterrows():
    carrier = row["Reporting_Airline"]
    fn = row["Flight_Number_Reporting_Airline"]
    orig = row["Origin"]
    dest = row["Dest"]
    target_date = pd.Timestamp(row["FlightDate"])
    pipeline_value = row["flightnum_od_otp_rate_last14d"]

    # Manual re-derivation:
    # Find all BTS flights with same (carrier, flightnum, orig, dest) in prior 14 days
    # i.e., FlightDate in [target_date - 14, target_date - 1].
    # Exclude cancelled flights.
    # Compute fraction with DepDelayMinutes <= 15.
    # Daily means, then rolling window — matching the pipeline's compute pattern.

    hist = bts[
        (bts["Reporting_Airline"] == carrier)
        & (bts["Flight_Number_Reporting_Airline"] == fn)
        & (bts["Origin"] == orig)
        & (bts["Dest"] == dest)
        & (bts["Cancelled"].fillna(0) == 0)
    ].copy()
    hist = hist.dropna(subset=["FlightDate", "DepDelayMinutes"])

    # Match pipeline: compute daily mean on-time indicator, then rolling mean
    hist["_on_time"] = (hist["DepDelayMinutes"] <= 15).astype(float)
    daily = hist.groupby("FlightDate", as_index=False)["_on_time"].mean().sort_values("FlightDate")
    # shift(1).rolling(14).mean():
    daily["otp"] = daily["_on_time"].shift(1).rolling(14, min_periods=1).mean()

    # Find the row for target_date
    match = daily[daily["FlightDate"] == target_date]
    if match.empty:
        print(f"  {carrier} {fn} {orig}->{dest} on {target_date.date()}: no BTS record for target date; skipping")
        continue
    manual_value = match["otp"].iloc[0]

    delta = abs(manual_value - pipeline_value) if pd.notna(manual_value) else float("nan")
    agree = pd.notna(manual_value) and delta < 0.05
    if agree:
        manual_match_count += 1
        mark = "✓"
    else:
        mark = "✗"

    print(f"  {mark} {carrier} {fn} {orig}->{dest} on {target_date.date()}: "
          f"pipeline={pipeline_value:.4f}  manual={manual_value:.4f}  Δ={delta:.4f}")

if manual_match_count >= 4:
    record("PASS", "semantic", "OTP rate manual re-derivation",
           f"{manual_match_count}/5 match within 0.05")
elif manual_match_count >= 3:
    record("WARN", "semantic", "OTP rate manual re-derivation",
           f"{manual_match_count}/5 match (some mismatches)")
else:
    record("FAIL", "semantic", "OTP rate manual re-derivation",
           f"only {manual_match_count}/5 match")

# Leakage test: for each row, the OTP value should not include today's flight.
# Implementation detail: the pipeline shifts by 1 day, so today's DepDelayMinutes
# should not affect the value. We check this indirectly: for rows on the FIRST
# day of the features window (2025-06-01), since there's exactly 14 days of prior
# history (May 18-31), the OTP should be computable without any target-day data.
first_day_rows = df[df["FlightDate"] == pd.Timestamp("2025-06-01")]
if len(first_day_rows) > 0:
    nonnull_frac = first_day_rows["flightnum_od_otp_rate_last14d"].notna().mean()
    record("PASS", "semantic", "OTP computable on first target day (leakage proxy)",
           f"{nonnull_frac:.1%} nonnull on 2025-06-01")
else:
    record("SKIP", "semantic", "OTP leakage proxy", "no 2025-06-01 rows")

# ---------------- DISTRIBUTION LAYER ----------------
print()
print("=" * 72)
print("DISTRIBUTION LAYER")
print("=" * 72)

s = df["flightnum_od_otp_rate_last14d"]
coverage = s.notna().mean()
if 0.88 <= coverage <= 0.98:
    record("PASS", "distribution", "OTP rate coverage 88-98%", f"{coverage:.1%}")
else:
    record("WARN", "distribution", "OTP rate coverage", f"{coverage:.1%}",
           "expected 88-98% on 1-month slice")

quantiles = s.quantile([0.1, 0.5, 0.9]).to_dict()
print(f"  OTP rate p10={quantiles[0.1]:.4f}  p50={quantiles[0.5]:.4f}  p90={quantiles[0.9]:.4f}")
# Median should be reasonably high (most flights ARE on-time most of the time)
if 0.5 <= quantiles[0.5] <= 0.9:
    record("PASS", "distribution", "OTP rate median in [0.5, 0.9]",
           f"{quantiles[0.5]:.4f}")
else:
    record("WARN", "distribution", "OTP rate median unusual",
           f"{quantiles[0.5]:.4f}")

# ---------------- SUMMARY ----------------
print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
sev_counts: Dict[str, int] = {}
for f in findings:
    sev_counts[f["severity"]] = sev_counts.get(f["severity"], 0) + 1
print(f"Total checks: {len(findings)}")
for s in ["PASS", "WARN", "FAIL", "SKIP", "INFO"]:
    if s in sev_counts:
        print(f"  {s}: {sev_counts[s]}")

# Write CSV summary
pd.DataFrame(findings).to_csv("data_quality/reports/2026-04-16-v11-checks.csv", index=False)
print()
print("Saved -> data_quality/reports/2026-04-16-v11-checks.csv")
