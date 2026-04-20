"""Unit tests for ISD visibility: CSV parsing + hour-floor merge semantics."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "fetch_prune"))

from collect_isd_visibility import SM_TO_M, MAX_VIS_M  # noqa: E402


def _parse_iem_csv(raw: str) -> pd.DataFrame:
    """Parse the IEM-format CSV the way collect_isd_visibility._fetch_one does.
    Extracted as a free function so tests don't need network."""
    df = pd.read_csv(io.StringIO(raw))
    df["valid_utc"] = pd.to_datetime(df["valid"], errors="coerce", utc=True)
    df["visibility_sm"] = pd.to_numeric(df["vsby"], errors="coerce")
    df["visibility_m"] = (df["visibility_sm"] * SM_TO_M).clip(lower=0.0, upper=MAX_VIS_M)
    df = df.dropna(subset=["valid_utc"])
    return df[["valid_utc", "visibility_sm", "visibility_m"]].sort_values("valid_utc").reset_index(drop=True)


class TestIEMParser:
    def test_happy_path_ten_mile_vis(self):
        raw = (
            "station,valid,vsby\n"
            "DFW,2025-03-01 00:53,10.00\n"
            "DFW,2025-03-01 01:53,5.00\n"
            "DFW,2025-03-01 02:53,0.25\n"
        )
        df = _parse_iem_csv(raw)
        assert len(df) == 3
        assert df["visibility_sm"].tolist() == [10.0, 5.0, 0.25]
        # 10 SM clipped at 25000 m → 16093.44 m (no clipping needed, 10 × 1609.344 < 25000)
        assert abs(df.loc[0, "visibility_m"] - 16093.44) < 1e-2
        assert abs(df.loc[2, "visibility_m"] - 0.25 * 1609.344) < 1e-2

    def test_missing_and_trace_become_nan(self):
        """IEM emits 'M' (missing) and 'T' (trace) tokens. Parsing must yield NaN,
        not strings, so downstream merges don't crash."""
        raw = (
            "station,valid,vsby\n"
            "DFW,2025-03-01 00:00,M\n"
            "DFW,2025-03-01 01:00,T\n"
            "DFW,2025-03-01 02:00,3.0\n"
        )
        df = _parse_iem_csv(raw)
        assert pd.isna(df.loc[0, "visibility_sm"])
        assert pd.isna(df.loc[0, "visibility_m"])
        assert df.loc[2, "visibility_sm"] == 3.0

    def test_high_vis_clipped_at_25km(self):
        """Some stations report >15 SM (e.g., 30.00). After conversion this is
        48,280 m which must clip to MAX_VIS_M for train/inference consistency
        with the existing Open-Meteo clip."""
        raw = (
            "station,valid,vsby\n"
            "DFW,2025-03-01 00:00,30.00\n"
        )
        df = _parse_iem_csv(raw)
        assert df.loc[0, "visibility_m"] == MAX_VIS_M

    def test_utc_roundtrip(self):
        """valid_utc must be timezone-aware UTC so the hour-floor merge with
        flight scheduled_dep_utc works without tz surprises."""
        raw = "station,valid,vsby\nDFW,2025-03-01 12:53,10.00\n"
        df = _parse_iem_csv(raw)
        assert df.loc[0, "valid_utc"].tz is not None
        assert df.loc[0, "valid_utc"].tz.utcoffset(None).total_seconds() == 0


class TestHourFloorMerge:
    """features_*.py joins ISD visibility to flights by hour-floor.
    Validate the join semantics in isolation since they're the #1 risk for
    silent wrong-value bugs."""

    @staticmethod
    def _make_vis(timestamps: list[str], sm: list[float]) -> pd.DataFrame:
        df = pd.DataFrame({
            "valid_utc": pd.to_datetime(timestamps, utc=True),
            "visibility_sm": sm,
            "visibility_m": [s * SM_TO_M for s in sm],
        })
        return df

    @staticmethod
    def _hour_floor_merge(flights: pd.DataFrame, vis: pd.DataFrame) -> pd.DataFrame:
        """Minimal hour-floor merge matching the production behavior."""
        f = flights.copy()
        v = vis.copy()
        f["dep_utc_hour"] = pd.to_datetime(f["dep_utc"], utc=True).dt.floor("h")
        v["dep_utc_hour"] = v["valid_utc"].dt.floor("h")
        # If multiple obs in same hour (e.g., SPECI), keep the earliest
        v = v.sort_values("valid_utc").drop_duplicates("dep_utc_hour", keep="first")
        return f.merge(v[["dep_utc_hour", "visibility_m"]], on="dep_utc_hour", how="left")

    def test_standard_metar_timestamp_matches_same_hour_flight(self):
        """METARs typically report at HH:53. A 15:30 UTC flight should match
        the 15:53 METAR (both floor to 15:00)."""
        vis = self._make_vis(["2025-03-01 15:53:00"], [3.0])
        flights = pd.DataFrame({"dep_utc": ["2025-03-01 15:30:00"]})
        out = self._hour_floor_merge(flights, vis)
        assert out.loc[0, "visibility_m"] == 3.0 * SM_TO_M

    def test_flight_just_before_metar_gets_previous_hour(self):
        """A 15:00 flight floors to 15:00, a 15:53 METAR floors to 15:00 — match.
        But a 14:55 flight floors to 14:00, which has no METAR → NaN."""
        vis = self._make_vis(["2025-03-01 15:53:00"], [3.0])
        flights = pd.DataFrame({
            "dep_utc": ["2025-03-01 15:00:00", "2025-03-01 14:55:00"],
        })
        out = self._hour_floor_merge(flights, vis)
        assert out.loc[0, "visibility_m"] == 3.0 * SM_TO_M
        assert pd.isna(out.loc[1, "visibility_m"])

    def test_multiple_obs_same_hour_dedupe_earliest_kept(self):
        """If a station issues a SPECI at 15:20 and a regular METAR at 15:53,
        both floor to 15:00. We keep the first (earliest) to match what the
        pilot would have seen at planning time — not the worst-case."""
        vis = self._make_vis(
            ["2025-03-01 15:20:00", "2025-03-01 15:53:00"],
            [0.5, 3.0],  # 0.5 SM is much worse than 3.0 SM
        )
        flights = pd.DataFrame({"dep_utc": ["2025-03-01 15:40:00"]})
        out = self._hour_floor_merge(flights, vis)
        # Earliest kept → 0.5 SM (not 3.0)
        assert abs(out.loc[0, "visibility_m"] - 0.5 * SM_TO_M) < 1e-2

    def test_nan_vis_does_not_break_merge(self):
        vis = self._make_vis(["2025-03-01 15:53:00"], [float("nan")])
        flights = pd.DataFrame({"dep_utc": ["2025-03-01 15:30:00"]})
        out = self._hour_floor_merge(flights, vis)
        assert pd.isna(out.loc[0, "visibility_m"])
