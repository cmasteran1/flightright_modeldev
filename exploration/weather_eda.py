#!/usr/bin/env python
"""Orchestrator -- runs all or selected weather EDA scripts."""

import argparse, importlib, json, sys, time
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

SCRIPTS = [
    "01_univariate",
    "02_correlation",
    "03_weathercode",
    "04_visibility_cape",
    "05_seasonal_geographic",
    "06_model_importance",
    "07_interactions",
    "08_airline_comparison",
    "10_full_feature_importance",
    "11_redundancy_analysis",
    "12_drop_candidates",
    "13_interaction_effects",
    "14_new_feature_candidates",
    "15_write_findings",
    "16_tail_aircraft_terminal_analysis",
    "17_tail_historical_tracking",
    "18_cascading_delay_risk",
]

# Scripts that don't take airline arg
NO_AIRLINE_SCRIPTS = {"08_airline_comparison", "15_write_findings"}


def main():
    parser = argparse.ArgumentParser(description="Weather-delay EDA orchestrator")
    parser.add_argument("--airline", default="WN", help="Airline code (default WN)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N rows for fast iteration")
    parser.add_argument("--scripts", default="all",
                        help="Comma-separated script names or 'all'")
    args = parser.parse_args()

    requested = SCRIPTS if args.scripts == "all" else args.scripts.split(",")
    requested = [s.strip() for s in requested]

    all_summaries = {}
    for script_name in requested:
        if script_name not in SCRIPTS:
            print(f"!!  Unknown script: {script_name}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"  Running {script_name} ...")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            mod = importlib.import_module(script_name)
            if script_name in NO_AIRLINE_SCRIPTS:
                result = mod.run(sample_n=args.sample) if "sample_n" in mod.run.__code__.co_varnames else mod.run()
            else:
                result = mod.run(airline=args.airline, sample_n=args.sample)
            all_summaries[script_name] = result or {}
        except Exception as e:
            print(f"  ERROR in {script_name}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[script_name] = {"error": str(e)}
        elapsed = time.time() - t0
        print(f"  [{script_name}] finished in {elapsed:.1f}s")

    # ── generate findings summary ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating findings summary ...")
    print(f"{'='*60}")

    lines = [
        f"# Weather Features Impact Analysis -- {args.airline}",
        f"",
        f"**Generated**: auto  ",
        f"**Sample size**: {'Full dataset' if not args.sample else f'{args.sample:,} rows'}",
        f"",
    ]

    # 01 summary
    s01 = all_summaries.get("01_univariate", {})
    if "top6_by_ks" in s01:
        lines += [
            "## 1. Univariate Analysis",
            f"Top features by distributional separation (KS statistic): "
            f"{', '.join(s01['top6_by_ks'])}",
            "",
        ]

    # 02 summary
    s02 = all_summaries.get("02_correlation", {})
    if "spearman" in s02:
        top_corr = sorted(s02["spearman"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        lines += [
            "## 2. Correlation Analysis",
            "Top-5 Spearman correlations with DepDelayMinutes:",
        ]
        for feat, val in top_corr:
            lines.append(f"  - {feat}: {val:.4f}")
        lines.append("")

    # 04 summary
    s04 = all_summaries.get("04_visibility_cape", {})
    if "cape_zero_fraction" in s04:
        lines += [
            "## 4. Visibility & CAPE",
            f"- CAPE = 0 fraction: {s04['cape_zero_fraction']:.1%}",
            f"- Delay rate CAPE=0: {s04['cape_eq0_delay_rate']:.3f}",
            f"- Delay rate CAPE>0: {s04['cape_gt0_delay_rate']:.3f}",
            "",
        ]

    # 06 summary
    s06 = all_summaries.get("06_model_importance", {})
    if "weather_only_auc" in s06:
        lines += [
            "## 6. Weather-Only Model",
            f"- **Weather-only AUC**: {s06['weather_only_auc']:.4f}",
            "",
        ]
        if "feature_importance" in s06:
            top5 = sorted(s06["feature_importance"].items(), key=lambda x: -x[1])[:5]
            lines.append("Top-5 feature importances:")
            for feat, val in top5:
                lines.append(f"  - {feat}: {val:.1f}")
            lines.append("")

    # 07 summary
    s07 = all_summaries.get("07_interactions", {})
    if "candidate_features" in s07:
        lines += [
            "## 7. Interaction Effects & Candidate Features",
            "Candidate engineered features (Pearson r with y_dep_ge15):",
        ]
        for feat, val in s07["candidate_features"].items():
            lines.append(f"  - {feat}: {val:.4f}")
        lines.append("")

    # 08 summary
    s08 = all_summaries.get("08_airline_comparison", {})
    if "weather_only_auc" in s08:
        lines += [
            "## 8. Airline Comparison",
            "Weather-only AUC by airline:",
        ]
        for al, auc in s08["weather_only_auc"].items():
            lines.append(f"  - {al}: {auc:.4f}")
        lines.append("")

    findings_path = Path(__file__).resolve().parent / "findings_summary.md"
    findings_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary written -> {findings_path}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
