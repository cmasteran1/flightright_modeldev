"""15 -- Generate comprehensive v7g feature analysis document."""

import json, sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / "figures"
OUT = Path(__file__).resolve().parent / "v7g_feature_analysis.md"


def _load(name):
    p = FIGURES / f"{name}_summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def run(**kwargs):
    print("[15] Generating v7g_feature_analysis.md ...")
    s10 = _load("10")
    s11 = _load("11")
    s12 = _load("12")
    s13 = _load("13")
    s14 = _load("14")
    # also pull from earlier weather EDA
    s06 = _load("06")
    s08 = _load("08")

    lines = []
    lines.append("# v7g Feature Analysis: Departure & Arrival Models")
    lines.append("")
    lines.append("## 1. Model Performance Overview")
    lines.append("")
    if "dep" in s10:
        lines.append(f"- **Departure model AUC**: {s10['dep']['auc']}")
    if "arr" in s10:
        lines.append(f"- **Arrival model AUC**: {s10['arr']['auc']}")
    if "weather_only_auc" in s06:
        lines.append(f"- **Weather-only model AUC**: {s06['weather_only_auc']} (baseline reference)")
    lines.append("")

    # -- Top features --
    lines.append("## 2. Feature Importance Rankings")
    lines.append("")
    if "dep" in s10 and "shap" in s10["dep"]:
        lines.append("### Departure Model -- Top 15 by SHAP")
        lines.append("")
        lines.append("| Rank | Feature | SHAP |")
        lines.append("|------|---------|------|")
        sorted_dep = sorted(s10["dep"]["shap"].items(), key=lambda x: -x[1])[:15]
        for i, (feat, val) in enumerate(sorted_dep, 1):
            lines.append(f"| {i} | {feat} | {val:.5f} |")
        lines.append("")

    if "arr" in s10 and "shap" in s10["arr"]:
        lines.append("### Arrival Model -- Top 15 by SHAP")
        lines.append("")
        lines.append("| Rank | Feature | SHAP |")
        lines.append("|------|---------|------|")
        sorted_arr = sorted(s10["arr"]["shap"].items(), key=lambda x: -x[1])[:15]
        for i, (feat, val) in enumerate(sorted_arr, 1):
            lines.append(f"| {i} | {feat} | {val:.5f} |")
        lines.append("")

    # -- Redundancy --
    lines.append("## 3. Redundancy Analysis")
    lines.append("")
    if "high_corr_pairs" in s11:
        n_pairs = len(s11["high_corr_pairs"])
        lines.append(f"Found **{n_pairs}** feature pairs with |Spearman r| > 0.85.")
        lines.append("")
        if n_pairs > 0:
            lines.append("### Highly Correlated Pairs (top 15)")
            lines.append("")
            lines.append("| Feature A | Feature B | r |")
            lines.append("|-----------|-----------|---|")
            for pair in s11["high_corr_pairs"][:15]:
                lines.append(f"| {pair['feat_a']} | {pair['feat_b']} | {pair['r']:.3f} |")
            lines.append("")

    if "rolling_groups" in s11:
        lines.append("### Rolling Window Redundancy")
        lines.append("")
        lines.append("Within each rolling window group (last1/7/14), features are highly correlated:")
        lines.append("")
        for name, info in s11["rolling_groups"].items():
            lines.append(f"- **{name}**: min cross-correlation = {info['min_r']:.3f}")
        lines.append("")
        lines.append("The last1 (1-day) window carries the most unique signal due to recency. ")
        lines.append("The last14 (14-day) window is the most stable but least reactive to sudden changes. ")
        lines.append("Consider keeping last1 and last14, dropping last7 where redundant.")
        lines.append("")

    # -- Drop recommendations --
    lines.append("## 4. Features Recommended to Drop")
    lines.append("")
    if "recommended_drops" in s12:
        drops = s12["recommended_drops"]
        lines.append(f"Of {s12.get('total_features', '?')} total features, "
                     f"**{len(drops)}** are candidates for removal.")
        lines.append("")
        if drops:
            lines.append("| Feature | Importance | SHAP | Reason |")
            lines.append("|---------|-----------|------|--------|")
            for d in drops[:20]:
                reasons = ", ".join(d["reasons"]) if isinstance(d["reasons"], list) else str(d["reasons"])
                lines.append(f"| {d['feature']} | {d['importance']:.1f} | {d['shap']:.5f} | {reasons} |")
            lines.append("")

    # -- Interactions --
    lines.append("## 5. Top Feature Interactions")
    lines.append("")
    if "top20_interactions" in s13:
        lines.append("CatBoost-detected pairwise interactions (internal splitting patterns):")
        lines.append("")
        lines.append("| Rank | Feature A | Feature B | Strength |")
        lines.append("|------|-----------|-----------|----------|")
        for i, inter in enumerate(s13["top20_interactions"][:10], 1):
            lines.append(f"| {i} | {inter['feat_a']} | {inter['feat_b']} | {inter['strength']:.1f} |")
        lines.append("")

    # -- New feature candidates --
    lines.append("## 6. Recommended New Features")
    lines.append("")
    if "candidate_results" in s14:
        lines.append(f"Baseline full-model AUC: {s14.get('baseline_auc', '?')}")
        lines.append("")
        lines.append("| Feature | Description | AUC Delta (bp) | Pearson r |")
        lines.append("|---------|-------------|----------------|-----------|")
        for r in s14["candidate_results"]:
            bp = r["delta_auc"] * 10000
            lines.append(f"| {r['feature']} | {r['description']} | {bp:+.1f} | {r['pearson_r']:.4f} |")
        lines.append("")
        positives = [r for r in s14["candidate_results"] if r["delta_auc"] > 0]
        if positives:
            lines.append("**Recommended to add** (positive AUC impact):")
            lines.append("")
            for r in positives:
                lines.append(f"- **{r['feature']}**: {r['description']} ({r['delta_auc']*10000:+.1f} bp)")
            lines.append("")

    # -- Tail / Aircraft / Terminal viability --
    s16 = _load("16")
    s17 = _load("17")
    s18 = _load("18")

    lines.append("## 7. New-API Feature Viability (Tail, Aircraft, Terminal)")
    lines.append("")
    lines.append("With the new API providing tail number, aircraft type, and terminal ")
    lines.append("assignments days ahead of departure, the following features become viable.")
    lines.append("")
    if "dep_results" in s16:
        lines.append("### Marginal AUC Impact (Departure Model)")
        lines.append("")
        lines.append("| Feature Group | Description | AUC Delta (bp) | # Features |")
        lines.append("|---------------|-------------|----------------|------------|")
        for r in s16["dep_results"]:
            lines.append(f"| {r['group']} | {r['desc'][:60]} | {r['delta_bp']:+.1f} | {r['n_features_added']} |")
        lines.append("")
        # Highlight the combined package
        combined = [r for r in s16["dep_results"] if r["group"] == "ALL_tail_features"]
        if combined:
            c = combined[0]
            lines.append(f"**Combined tail/aircraft package**: {c['delta_bp']:+.1f} bp "
                         f"({c['n_features_added']} features)")
            lines.append("")

    if "terminal_analysis" in s16:
        ta = s16["terminal_analysis"]
        lines.append("### Terminal Features")
        lines.append("")
        lines.append(f"**Status**: {ta['status']}")
        lines.append("")
        lines.append(ta["note"])
        lines.append("")
        lines.append(f"**Recommendation**: {ta['recommendation']}")
        lines.append("")

    # -- Tail historical tracking --
    lines.append("## 8. Tail-Based Historical Tracking")
    lines.append("")
    if "tail_marginal_beyond_flightnum" in s17:
        tm = s17["tail_marginal_beyond_flightnum"]
        lines.append(f"- Tail history marginal value beyond flight-number history: "
                     f"**{tm['delta_bp']:+.1f} bp**")
        lines.append(f"  - Base AUC (with flightnum): {tm['base_auc']}")
        lines.append(f"  - With tail history added: {tm['with_tail_auc']}")
        lines.append("")
    if "tail_by_window" in s17:
        lines.append("### Window-by-Window Tail Importance")
        lines.append("")
        lines.append("| Window | AUC Delta (bp) |")
        lines.append("|--------|----------------|")
        for w in s17["tail_by_window"]:
            lines.append(f"| {w['window']} | {w['delta_bp']:+.1f} |")
        lines.append("")
    if "tail_persistence" in s17:
        tp = s17["tail_persistence"]
        lines.append(f"### Tail Delay Persistence: r = {tp['pearson_r_first_vs_second_half']:.3f}")
        lines.append("")
        lines.append(tp["interpretation"])
        lines.append("")
    if "correlation_with_target" in s17:
        lines.append("### History Signal Correlation Comparison")
        lines.append("")
        lines.append("| Source | last1 | last7 | last14 |")
        lines.append("|--------|-------|-------|--------|")
        for name, corrs in s17["correlation_with_target"].items():
            vals = [f"{corrs.get(w, 0):.4f}" for w in ["1", "7", "14"]]
            lines.append(f"| {name} | {' | '.join(vals)} |")
        lines.append("")

    # -- Cascading delay risk --
    lines.append("## 9. Cascading Delay Risk")
    lines.append("")
    if "daily_stats" in s18:
        ds = s18["daily_stats"]
        lines.append(f"- **{ds['total_tail_days']:,}** unique tail-days analyzed")
        lines.append(f"- Average legs per tail-day: **{ds['mean_legs_per_day']:.1f}**")
        lines.append(f"- Any delay: **{ds['pct_any_delay']:.1%}** of tail-days")
        lines.append(f"- Cascade (>1 leg delayed): **{ds['pct_cascade']:.1%}** of tail-days")
        if ds.get("pct_cascade_given_first_delayed"):
            lines.append(f"- **P(cascade | first leg delayed) = {ds['pct_cascade_given_first_delayed']:.1%}**")
        lines.append("")
    if "propagation" in s18:
        pr = s18["propagation"]
        lines.append("### Delay Propagation")
        lines.append("")
        lines.append(f"- P(delay | previous leg delayed) = **{pr['P_delay_given_prev_delayed']:.3f}**")
        lines.append(f"- P(delay | previous leg OK) = **{pr['P_delay_given_prev_ok']:.3f}**")
        lines.append(f"- **Lift factor: {pr['lift_factor']}x**")
        lines.append("")
    if "absorption_by_turn_time" in s18:
        lines.append("### Delay Absorption by Turn Time")
        lines.append("")
        lines.append("| Turn Time | P(next delayed | prev delayed) |")
        lines.append("|-----------|-------------------------------|")
        for turn, rate in s18["absorption_by_turn_time"].items():
            lines.append(f"| {turn} | {rate:.3f} |")
        lines.append("")
    if "cascade_prediction" in s18:
        cp = s18["cascade_prediction"]
        lines.append("### Cascade Prediction Model (schedule-only features)")
        lines.append("")
        lines.append(f"Using only features knowable days ahead (# legs, min turn time, etc.):")
        lines.append(f"- **AUC: {cp['auc']:.4f}**, Average Precision: {cp['average_precision']:.4f}")
        lines.append(f"- Test cascade rate: {cp['cascade_rate_test']:.1%}")
        lines.append("")
    if "proposed_cascade_features" in s18:
        lines.append("### Proposed Cascade Risk Features")
        lines.append("")
        lines.append("Features computable days ahead from tail number + schedule:")
        lines.append("")
        for name, desc in s18["proposed_cascade_features"].items():
            lines.append(f"- **`{name}`**: {desc}")
        lines.append("")
    if "cascade_feature_importance" in s18:
        lines.append("### Cascade Model Feature Importance")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("|---------|-----------|")
        sorted_ci = sorted(s18["cascade_feature_importance"].items(), key=lambda x: -x[1])
        for feat, imp in sorted_ci:
            lines.append(f"| {feat} | {imp:.1f} |")
        lines.append("")

    # -- Summary recommendations --
    lines.append("## 10. Summary of Recommendations")
    lines.append("")
    lines.append("### Add")
    lines.append("")
    if "candidate_results" in s14:
        for r in s14["candidate_results"]:
            if r["delta_auc"] > 0:
                lines.append(f"- `{r['feature']}`: {r['description']}")
    lines.append("")
    lines.append("### Drop (low importance or redundant)")
    lines.append("")
    if "recommended_drops" in s12:
        for d in s12["recommended_drops"][:10]:
            reasons = ", ".join(d["reasons"]) if isinstance(d["reasons"], list) else str(d["reasons"])
            lines.append(f"- `{d['feature']}`: {reasons}")
    lines.append("")
    lines.append("### Add (Tail/Aircraft -- new API)")
    lines.append("")
    lines.append("- `aircraft_type`: strongest single categorical addition")
    lines.append("- `has_recent_arrival_turn_5h`: strong binary signal for quick-turn risk")
    lines.append("- `turn_time_hours`: turnaround buffer signal")
    lines.append("- `tail_leg_num_day`: later legs carry more cascade risk")
    lines.append("- Full tail/aircraft package: best combined AUC gain")
    lines.append("")
    lines.append("### Add (Cascade Risk -- new API)")
    lines.append("")
    lines.append("- `tail_n_legs_scheduled`: more legs = more cascade opportunity")
    lines.append("- `tail_min_turn_time`: tightest turn in rotation = weakest link")
    lines.append("- `tail_has_tight_turn`: binary cascade vulnerability flag")
    lines.append("- `tail_route_complexity`: # unique airports in rotation")
    lines.append("")
    lines.append("### Keep Investigating")
    lines.append("")
    lines.append("- Rolling window consolidation: test dropping last7 across all groups")
    lines.append("- Hub spillover: test aggregating hub_0-4 into single max/mean features")
    lines.append("- Arrival model: destination weather features need same deep-dive as origin weather")
    lines.append("- Terminal features: evaluate when new API data available")
    lines.append("- Tail history rolling windows: currently marginal individually, "
                 "but valuable in the combined package")
    lines.append("")
    lines.append("---")
    lines.append("*Analysis run on WN (Southwest Airlines) data. See exploration/figures/ for all charts.*")

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(f"[15] Written -> {OUT}")
    print("[15] Done.")
    return {}


if __name__ == "__main__":
    run()
