"""18 -- Cascading delay risk analysis: given tail number days ahead,
       how can we model propagation risk through the daily rotation?

Key questions:
1. How much does a delayed first leg affect subsequent legs?
2. Can we predict cascading risk from tail schedule characteristics?
3. What engineered features would capture cascade risk?
"""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (set_plot_style, save_fig, short_name, FIGURES, DATA_ROOT,
                   DEP_FEATURE_ORDER, DEP_CAT_FEATURES, train_catboost_full)


def _load(airline, sample_n):
    cols = [
        "FlightDate", "Tail_Number", "Origin", "Dest",
        "CRSDepTime", "CRSElapsedTime", "Distance",
        "DepDelayMinutes",
        "y_dep_ge15", "y_dep_ge30", "y_dep_ge60",
        "tail_leg_num_day", "turn_time_hours",
        "sched_dep_hour", "dep_dow",
        "aircraft_type", "Aircraft_Age_Bucket",
        "tail_depdelay_mean_last1", "tail_depdelay_mean_last7", "tail_depdelay_mean_last14",
        "tail_lateaircraft_rate_last1", "tail_lateaircraft_rate_last7", "tail_lateaircraft_rate_last14",
        "origin_congestion_3h_total",
    ]
    path = DATA_ROOT / f"features_dep_{airline}_100_v7g_unbalanced.parquet"
    df = pd.read_parquet(path, columns=[c for c in cols])
    if sample_n and len(df) > sample_n:
        # For cascade analysis we need full days of tails -- sample by date instead
        dates = sorted(df["FlightDate"].unique())
        n_dates = max(1, int(len(dates) * sample_n / len(df)))
        rng = np.random.default_rng(42)
        keep_dates = set(rng.choice(dates, size=n_dates, replace=False))
        df = df[df["FlightDate"].isin(keep_dates)]
    return df


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[18] Loading data for cascade analysis ...")
    df = _load(airline, sample_n)
    print(f"[18] {len(df):,} rows, {df['Tail_Number'].nunique()} unique tails")
    summary = {}
    target = "y_dep_ge15"

    # ── 1. Build daily rotation sequences per tail ────────────────────
    print("[18] Building tail daily rotations ...")
    df = df.sort_values(["FlightDate", "Tail_Number", "CRSDepTime"])
    df["delayed"] = df[target].astype(float)

    # For each tail-day, get the sequence of delays
    daily = df.groupby(["FlightDate", "Tail_Number"]).agg(
        n_legs=("delayed", "count"),
        first_delayed=("delayed", "first"),
        total_delayed=("delayed", "sum"),
        max_delay_min=("DepDelayMinutes", "max"),
        mean_delay_min=("DepDelayMinutes", "mean"),
    ).reset_index()
    daily["any_delayed"] = (daily["total_delayed"] > 0).astype(int)
    daily["cascade"] = (daily["total_delayed"] > 1).astype(int)
    daily["cascade_given_first"] = np.nan
    mask = daily["first_delayed"] == 1
    daily.loc[mask, "cascade_given_first"] = (daily.loc[mask, "total_delayed"] > 1).astype(int)

    summary["daily_stats"] = {
        "total_tail_days": len(daily),
        "mean_legs_per_day": round(float(daily["n_legs"].mean()), 2),
        "pct_any_delay": round(float(daily["any_delayed"].mean()), 4),
        "pct_cascade": round(float(daily["cascade"].mean()), 4),
        "pct_cascade_given_first_delayed": round(float(daily.loc[mask, "cascade_given_first"].mean()), 4)
            if mask.sum() > 0 else None,
    }
    print(f"  Any delay: {daily['any_delayed'].mean():.1%}")
    print(f"  Cascade (>1 leg delayed): {daily['cascade'].mean():.1%}")
    cg = daily.loc[mask, "cascade_given_first"].mean() if mask.sum() > 0 else 0
    print(f"  Cascade given first delayed: {cg:.1%}")

    # ── 2. Propagation matrix: P(leg N delayed | leg N-1 delayed) ────
    print("[18] Computing propagation matrix ...")
    df_sorted = df.sort_values(["FlightDate", "Tail_Number", "CRSDepTime"])
    df_sorted["prev_delayed"] = df_sorted.groupby(["FlightDate", "Tail_Number"])["delayed"].shift(1)
    df_sorted["prev_delay_min"] = df_sorted.groupby(["FlightDate", "Tail_Number"])["DepDelayMinutes"].shift(1)

    prop = df_sorted.dropna(subset=["prev_delayed"])
    p_delay_if_prev_delayed = prop[prop["prev_delayed"] == 1]["delayed"].mean()
    p_delay_if_prev_ok = prop[prop["prev_delayed"] == 0]["delayed"].mean()
    summary["propagation"] = {
        "P_delay_given_prev_delayed": round(float(p_delay_if_prev_delayed), 4),
        "P_delay_given_prev_ok": round(float(p_delay_if_prev_ok), 4),
        "lift_factor": round(float(p_delay_if_prev_delayed / max(p_delay_if_prev_ok, 0.001)), 2),
    }
    print(f"  P(delay | prev delayed) = {p_delay_if_prev_delayed:.3f}")
    print(f"  P(delay | prev OK) = {p_delay_if_prev_ok:.3f}")
    print(f"  Lift = {p_delay_if_prev_delayed / max(p_delay_if_prev_ok, 0.001):.1f}x")

    # ── 3. Delay absorption: how much turn time needed to recover? ───
    print("[18] Delay absorption by turn time ...")
    valid_turn = prop[(prop["prev_delayed"] == 1) & prop["turn_time_hours"].notna()
                      & (prop["turn_time_hours"] > 0)].copy()
    if len(valid_turn) > 100:
        try:
            valid_turn["turn_bin"] = pd.cut(valid_turn["turn_time_hours"],
                                            bins=[0, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 24.0],
                                            labels=["<30m", "30-45m", "45-60m", "1-1.5h",
                                                    "1.5-2h", "2-4h", "4h+"])
            abs_grp = valid_turn.groupby("turn_bin", observed=True).agg(
                cascade_rate=("delayed", "mean"),
                n=("delayed", "count"),
            )
            abs_grp = abs_grp[abs_grp["n"] > 30]
            summary["absorption_by_turn_time"] = abs_grp["cascade_rate"].to_dict()
        except Exception:
            abs_grp = pd.DataFrame()
    else:
        abs_grp = pd.DataFrame()

    # ── 4. Cascade risk features that could be computed days ahead ────
    print("[18] Engineering cascade risk features ...")
    # Given we know tail number + schedule days ahead, we can compute:
    cascade_features = {
        "tail_n_legs_scheduled": (
            "Number of legs this tail is scheduled for today. "
            "More legs = more cascade opportunity."
        ),
        "tail_min_turn_time": (
            "Minimum scheduled turn time for this tail today. "
            "Tight turns have less recovery buffer."
        ),
        "tail_mean_turn_time": (
            "Average scheduled turn time. Loose schedules absorb delays better."
        ),
        "tail_has_tight_turn": (
            "Binary: any turn < 45 min in today's rotation. "
            "Single tight turn can cascade the rest of the day."
        ),
        "tail_first_dep_hour": (
            "First scheduled departure hour. Early starts accumulate more legs."
        ),
        "tail_route_complexity": (
            "Number of unique airports in today's rotation. "
            "More airports = more weather/congestion exposure."
        ),
        "tail_historical_delay_propensity": (
            "Tail's 14-day rolling delay rate. Some aircraft consistently delay "
            "(maintenance issues, configuration problems)."
        ),
        "tail_schedule_buffer_ratio": (
            "Total scheduled block time / total actual block time (historical). "
            "Tight scheduling leaves less slack."
        ),
    }
    summary["proposed_cascade_features"] = cascade_features

    # ── 5. Plots ──────────────────────────────────────────────────────
    print("[18] Generating plots ...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # a) Cascade rate by # legs
    ax = axes[0, 0]
    leg_grp = daily.groupby("n_legs").agg(
        cascade_rate=("cascade", "mean"), n=("cascade", "count"))
    leg_grp = leg_grp[leg_grp["n"] > 50]
    ax.bar(leg_grp.index, leg_grp["cascade_rate"], color="#D32F2F", alpha=0.85)
    ax.set_xlabel("# Legs Per Tail-Day")
    ax.set_ylabel("P(cascade)")
    ax.set_title("Cascade Rate by Daily Leg Count")

    # b) Propagation: P(delay | prev_delayed) vs P(delay | prev_ok)
    ax = axes[0, 1]
    bars = ax.bar(["Prev OK", "Prev Delayed"],
                  [p_delay_if_prev_ok, p_delay_if_prev_delayed],
                  color=["#4CAF50", "#D32F2F"], alpha=0.85)
    ax.set_ylabel("P(current leg delayed)")
    ax.set_title(f"Delay Propagation (Lift = {p_delay_if_prev_delayed/max(p_delay_if_prev_ok,0.001):.1f}x)")
    for bar, val in zip(bars, [p_delay_if_prev_ok, p_delay_if_prev_delayed]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")

    # c) Absorption by turn time
    ax = axes[0, 2]
    if not abs_grp.empty:
        ax.bar(range(len(abs_grp)), abs_grp["cascade_rate"], color="#FF9800", alpha=0.85)
        ax.set_xticks(range(len(abs_grp)))
        ax.set_xticklabels(abs_grp.index, fontsize=8, rotation=30)
        ax.set_ylabel("P(next leg delayed | prev delayed)")
        ax.set_title("Delay Absorption by Turn Time")
    else:
        ax.set_title("Insufficient turn time data")

    # d) Delay progression through the day
    ax = axes[1, 0]
    leg_delay = df.groupby("tail_leg_num_day").agg(
        delay_rate=(target, "mean"), mean_delay=("DepDelayMinutes", "mean"),
        n=(target, "count")).reset_index()
    leg_delay = leg_delay[leg_delay["n"] > 200]
    ax.plot(leg_delay["tail_leg_num_day"], leg_delay["delay_rate"],
            "o-", color="#D32F2F", markersize=8, label="P(delay >= 15)")
    ax.set_xlabel("Leg # of Day")
    ax.set_ylabel("Delay Rate")
    ax.set_title("Delay Rate Progression Through Daily Legs")
    ax.legend()

    # e) Previous delay magnitude vs cascade probability
    ax = axes[1, 1]
    if "prev_delay_min" in prop.columns:
        delayed_prev = prop[prop["prev_delayed"] == 1].copy()
        if len(delayed_prev) > 500:
            try:
                delayed_prev["delay_bin"] = pd.cut(delayed_prev["prev_delay_min"],
                    bins=[0, 15, 30, 45, 60, 120, 500],
                    labels=["0-15", "15-30", "30-45", "45-60", "60-120", "120+"])
                cascade_by_severity = delayed_prev.groupby("delay_bin", observed=True)["delayed"].agg(["mean", "count"])
                cascade_by_severity = cascade_by_severity[cascade_by_severity["count"] > 30]
                ax.bar(range(len(cascade_by_severity)), cascade_by_severity["mean"],
                       color="#7E57C2", alpha=0.85)
                ax.set_xticks(range(len(cascade_by_severity)))
                ax.set_xticklabels(cascade_by_severity.index, fontsize=9)
                ax.set_xlabel("Previous Leg Delay (min)")
                ax.set_ylabel("P(next leg delayed)")
                ax.set_title("Cascade Risk by Delay Severity")
            except Exception:
                ax.set_title("Cascade by severity (error)")

    # f) Proposed feature summary
    ax = axes[1, 2]
    ax.axis("off")
    feature_text = "PROPOSED CASCADE FEATURES\n(computable days ahead from schedule)\n\n"
    for i, (name, desc) in enumerate(list(cascade_features.items())[:6]):
        feature_text += f"{i+1}. {name}\n   {desc[:60]}...\n\n"
    ax.text(0.05, 0.95, feature_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"Cascading Delay Risk Analysis ({airline})", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_18a_cascading_delay_risk")

    # ── 6. Quantify cascading with CatBoost: can we predict cascade risk? ──
    print("[18] Training cascade prediction model ...")
    # Build a dataset: one row per tail-day with features knowable days ahead
    cascade_data = daily[daily["n_legs"] >= 2].copy()
    # Merge tail characteristics
    tail_chars = df.groupby(["FlightDate", "Tail_Number"]).agg(
        mean_turn=("turn_time_hours", "mean"),
        min_sched_dep=("CRSDepTime", "min"),
        max_sched_dep=("CRSDepTime", "max"),
        n_airports=("Origin", "nunique"),
        mean_distance=("Distance", "mean"),
    ).reset_index()
    cascade_data = cascade_data.merge(tail_chars, on=["FlightDate", "Tail_Number"], how="left")
    cascade_data = cascade_data.dropna(subset=["cascade"])

    # Features knowable days ahead
    ahead_features = ["n_legs", "mean_turn", "min_sched_dep", "max_sched_dep",
                      "n_airports", "mean_distance"]
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import roc_auc_score, average_precision_score

    cascade_data_valid = cascade_data.dropna(subset=ahead_features + ["cascade"])
    if len(cascade_data_valid) > 1000:
        dates = sorted(cascade_data_valid["FlightDate"].unique())
        split = int(len(dates) * 0.8)
        train_m = cascade_data_valid[cascade_data_valid["FlightDate"].isin(set(dates[:split]))]
        test_m = cascade_data_valid[cascade_data_valid["FlightDate"].isin(set(dates[split:]))]

        X_tr, y_tr = train_m[ahead_features], train_m["cascade"].astype(int)
        X_te, y_te = test_m[ahead_features], test_m["cascade"].astype(int)

        clf = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                 verbose=0, random_seed=42)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        cascade_auc = roc_auc_score(y_te, y_prob)
        cascade_ap = average_precision_score(y_te, y_prob)
        print(f"  Cascade prediction: AUC={cascade_auc:.4f}, AP={cascade_ap:.4f}")
        summary["cascade_prediction"] = {
            "auc": round(cascade_auc, 4),
            "average_precision": round(cascade_ap, 4),
            "features_used": ahead_features,
            "train_size": len(train_m),
            "test_size": len(test_m),
            "cascade_rate_test": round(float(y_te.mean()), 4),
        }
        # Feature importance for cascade model
        imp = clf.get_feature_importance()
        summary["cascade_feature_importance"] = {
            ahead_features[i]: round(float(imp[i]), 1) for i in range(len(ahead_features))
        }

    with open(FIGURES / "18_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[18] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
