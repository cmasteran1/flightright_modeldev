"""Shared helpers for weather-delay EDA."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "flightrightdata" / "data" / "processed"
FIGURES   = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── weather feature columns ────────────────────────────────────────────
DAILY_WEATHER = [
    "origin_temp_max_K", "origin_temp_min_K",
    "origin_daily_precip_sum_mm",
    "origin_daily_windspeed_max_kmh", "origin_daily_windgusts_max_kmh",
    "origin_daily_weathercode",
]
HOURLY_WEATHER = [
    "origin_dep_temp_K", "origin_dep_precip_mm",
    "origin_dep_windspeed_kmh", "origin_dep_windgusts_kmh",
    "origin_dep_visibility_m", "origin_dep_cape_jkg",
    "origin_dep_log1p_cape", "origin_dep_cloudcover_pct",
    "origin_dep_hour_weathercode",
]
NUMERIC_WEATHER = [c for c in DAILY_WEATHER + HOURLY_WEATHER
                   if "weathercode" not in c]
CAT_WEATHER = ["origin_daily_weathercode", "origin_dep_hour_weathercode"]
ALL_WEATHER = DAILY_WEATHER + HOURLY_WEATHER

TARGETS = ["DepDelayMinutes", "y_dep_ge15", "y_dep_ge30", "y_dep_ge45", "y_dep_ge60"]
CONTEXT = ["FlightDate", "Origin", "Dest", "Reporting_Airline",
           "sched_dep_hour", "dep_dow", "flight_month"]

# ── WMO weather-code mapping ──────────────────────────────────────────
WMO_CODE_MAP = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ slight hail", 99: "Thunderstorm w/ heavy hail",
}

WMO_SEVERITY_MAP = {
    0: "Clear", 1: "Clear", 2: "Clear", 3: "Overcast",
    45: "Fog", 48: "Fog",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
    56: "Freezing Precip", 57: "Freezing Precip",
    61: "Rain", 63: "Rain", 65: "Rain",
    66: "Freezing Precip", 67: "Freezing Precip",
    71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
    80: "Rain Showers", 81: "Rain Showers", 82: "Rain Showers",
    85: "Snow Showers", 86: "Snow Showers",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}

SEVERITY_ORDER = ["Clear", "Overcast", "Fog", "Drizzle", "Rain",
                  "Rain Showers", "Freezing Precip", "Snow",
                  "Snow Showers", "Thunderstorm"]

SEVERITY_COLORS = {
    "Clear": "#4CAF50", "Overcast": "#9E9E9E", "Fog": "#CE93D8",
    "Drizzle": "#64B5F6", "Rain": "#1565C0", "Rain Showers": "#0D47A1",
    "Freezing Precip": "#E91E63", "Snow": "#00BCD4",
    "Snow Showers": "#00838F", "Thunderstorm": "#FF5722",
}

# ── US region mapping (covers WN top-100 + major airports) ────────────
_NE = ["BOS","BDL","PVD","BWI","DCA","IAD","PHL","EWR","JFK","LGA",
       "SYR","BUF","ROC","ALB","ISP","PIT","RIC","ORF","SWF","HPN"]
_SE = ["ATL","CLT","RDU","MYR","CHS","SAV","JAX","MCO","TPA","FLL",
       "MIA","PBI","RSW","SRQ","MSY","BNA","MEM","SDF","GSP","CAK",
       "PNS","TYS","BHM","HSV","GSO","DAB","PIE"]
_MW = ["ORD","MDW","DTW","CLE","CMH","IND","MKE","MSP","STL","MCI",
       "DSM","OMA","GRR","FWA","CVG","DAY","ICT","LIT","TUL","OKC","XNA"]
_SW = ["DFW","DAL","IAH","HOU","SAT","AUS","ELP","ABQ","PHX","TUS","LBB","MAF","MSY"]
_WE = ["LAX","SFO","SJC","OAK","SAN","SMF","ONT","BUR","LGB","SNA",
       "SEA","PDX","BOI","RNO","LAS","DEN","SLC","COS","ANC","HNL",
       "OGG","LIH","KOA"]

US_REGION_MAP = {}
for _codes, _region in [(_NE, "Northeast"), (_SE, "Southeast"),
                         (_MW, "Midwest"), (_SW, "Southwest"), (_WE, "West")]:
    for c in _codes:
        US_REGION_MAP[c] = _region


# ── data loading ───────────────────────────────────────────────────────
def load_weather_df(airline="WN", sample_n=None, extra_cols=None):
    """Load weather + target + context columns from processed parquet."""
    cols = list(set(ALL_WEATHER + TARGETS + CONTEXT + (extra_cols or [])))
    path = DATA_ROOT / f"features_dep_{airline}_100_v7g_unbalanced.parquet"
    df = pd.read_parquet(path, columns=cols)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df


# ── plot helpers ───────────────────────────────────────────────────────
def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.figsize": (12, 8),
    })

def save_fig(fig, name):
    out = FIGURES / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}")
    return out


def safe_imshow(ax, data, **kwargs):
    """imshow that handles NaN/object/NAType from unstacked DataFrames."""
    df = pd.DataFrame(data).fillna(0.0)
    arr = df.to_numpy(dtype=float, na_value=0.0)
    return ax.imshow(arr, **kwargs)


def short_name(col):
    """Shorten column name for chart labels."""
    return (col
            .replace("origin_daily_", "daily_")
            .replace("origin_dep_", "dep_")
            .replace("origin_", ""))


# ── v7g resolved feature lists (from trained WN models) ───────────────
DEP_CAT_FEATURES = [
    "Reporting_Airline", "Origin", "Dest", "DepTimeBlk",
    "origin_dep_hour_weathercode",
]
DEP_NUM_FEATURES = [
    "flightnum_od_depdelay_mean_last1", "flightnum_od_depdelay_mean_last7",
    "flightnum_od_depdelay_mean_last14", "flightnum_od_depdelay_median_last1",
    "flightnum_od_depdelay_median_last7", "flightnum_od_depdelay_median_last14",
    "flightnum_od_support_count_last14d", "flightnum_od_low_support_last14d",
    "carrier_depdelay_mean_last1", "carrier_depdelay_mean_last7",
    "carrier_depdelay_mean_last14",
    "carrier_origin_depdelay_mean_last1", "carrier_origin_depdelay_mean_last7",
    "carrier_origin_depdelay_mean_last14",
    "origin_depdelay_mean_last1", "origin_depdelay_mean_last7",
    "origin_depdelay_mean_last14",
    "origin_lateaircraft_rate_last1", "origin_lateaircraft_rate_last7",
    "origin_lateaircraft_rate_last14",
    "carrier_lateaircraft_rate_last1", "carrier_lateaircraft_rate_last7",
    "carrier_lateaircraft_rate_last14",
    "hub_0_depdelay_mean_last1", "hub_0_depdelay_mean_last7",
    "hub_0_depdelay_mean_last14",
    "hub_0_lateaircraft_rate_last1", "hub_0_lateaircraft_rate_last7",
    "hub_0_lateaircraft_rate_last14",
    "hub_1_depdelay_mean_last1", "hub_1_depdelay_mean_last7",
    "hub_1_depdelay_mean_last14",
    "hub_1_lateaircraft_rate_last1", "hub_1_lateaircraft_rate_last7",
    "hub_1_lateaircraft_rate_last14",
    "hub_2_depdelay_mean_last1", "hub_2_depdelay_mean_last7",
    "hub_2_depdelay_mean_last14",
    "hub_2_lateaircraft_rate_last1", "hub_2_lateaircraft_rate_last7",
    "hub_2_lateaircraft_rate_last14",
    "hub_3_depdelay_mean_last1", "hub_3_depdelay_mean_last7",
    "hub_3_depdelay_mean_last14",
    "hub_3_lateaircraft_rate_last1", "hub_3_lateaircraft_rate_last7",
    "hub_3_lateaircraft_rate_last14",
    "hub_4_depdelay_mean_last1", "hub_4_depdelay_mean_last7",
    "hub_4_depdelay_mean_last14",
    "hub_4_lateaircraft_rate_last1", "hub_4_lateaircraft_rate_last7",
    "hub_4_lateaircraft_rate_last14",
    "dest_depdelay_mean_last1", "dest_depdelay_mean_last7",
    "dest_depdelay_mean_last14",
    "dest_lateaircraft_rate_last1", "dest_lateaircraft_rate_last7",
    "dest_lateaircraft_rate_last14",
    "origin_congestion_3h_total", "origin_airline_congestion_3h_total",
    "dep_dow", "sched_dep_hour", "is_holiday",
    "CRSDepTime", "CRSElapsedTime", "Distance",
    "origin_temp_max_K", "origin_temp_min_K", "origin_daily_precip_sum_mm",
    "origin_daily_windspeed_max_kmh", "origin_daily_windgusts_max_kmh",
    "origin_dep_temp_K", "origin_dep_precip_mm", "origin_dep_windspeed_kmh",
    "origin_dep_windgusts_kmh", "origin_dep_visibility_m",
    "origin_dep_cape_jkg", "origin_dep_log1p_cape", "origin_dep_cloudcover_pct",
]
DEP_FEATURE_ORDER = DEP_CAT_FEATURES + DEP_NUM_FEATURES

ARR_CAT_FEATURES = [
    "Reporting_Airline", "Origin", "Dest", "DepTimeBlk",
    "origin_dep_hour_weathercode",
]
ARR_NUM_FEATURES = [
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance",
    "arrdelay_mean_7d_fn_od", "arrdelay_n_7d_fn_od",
    "arrdelay_mean_14d_fn_od", "arrdelay_n_14d_fn_od",
    "arrdelay_median_7d_fn_od",
    "arrdelay_mean_7d_car_od", "arrdelay_n_7d_car_od",
    "arrdelay_mean_14d_car_od", "arrdelay_n_14d_car_od",
    "arrdelay_median_7d_car_od",
    "airtime_mean_7d_fn_od", "airtime_n_7d_fn_od",
    "airtime_mean_14d_fn_od", "airtime_n_14d_fn_od",
    "airtime_mean_7d_car_od", "airtime_n_7d_car_od",
    "airtime_mean_14d_car_od", "airtime_n_14d_car_od",
    "wo_slip_mean_7d_fn_od", "wo_slip_n_7d_fn_od",
    "wo_slip_mean_14d_fn_od", "wo_slip_n_14d_fn_od",
    "wo_slip_mean_7d_origin_blk", "wo_slip_n_7d_origin_blk",
    "wo_slip_mean_14d_origin_blk", "wo_slip_n_14d_origin_blk",
    "dest_arrivals_pm60_sched", "dest_airline_arrivals_pm60_sched",
    "dest_arrivals_pm60_eta", "dest_airline_arrivals_pm60_eta",
    "dep_dow", "sched_dep_hour", "sched_arr_hour",
    "dest_arr_temperature_2m", "dest_arr_precipitation",
    "dest_arr_windspeed_10m", "dest_arr_windgusts_10m",
    "dest_arr_visibility", "dest_arr_cape", "dest_arr_cloudcover",
    "flightnum_od_depdelay_mean_last1", "flightnum_od_depdelay_mean_last7",
    "flightnum_od_depdelay_mean_last14",
    "flightnum_od_depdelay_median_last1", "flightnum_od_depdelay_median_last7",
    "flightnum_od_depdelay_median_last14",
    "flightnum_od_support_count_last14d", "flightnum_od_low_support_last14d",
    "carrier_depdelay_mean_last1", "carrier_depdelay_mean_last7",
    "carrier_depdelay_mean_last14",
    "carrier_origin_depdelay_mean_last1", "carrier_origin_depdelay_mean_last7",
    "carrier_origin_depdelay_mean_last14",
    "origin_depdelay_mean_last1", "origin_depdelay_mean_last7",
    "origin_depdelay_mean_last14",
    "origin_lateaircraft_rate_last1", "origin_lateaircraft_rate_last7",
    "origin_lateaircraft_rate_last14",
    "carrier_lateaircraft_rate_last1", "carrier_lateaircraft_rate_last7",
    "carrier_lateaircraft_rate_last14",
    "hub_0_depdelay_mean_last1", "hub_0_depdelay_mean_last7",
    "hub_0_depdelay_mean_last14",
    "hub_0_lateaircraft_rate_last1", "hub_0_lateaircraft_rate_last7",
    "hub_0_lateaircraft_rate_last14",
    "hub_1_depdelay_mean_last1", "hub_1_depdelay_mean_last7",
    "hub_1_depdelay_mean_last14",
    "hub_1_lateaircraft_rate_last1", "hub_1_lateaircraft_rate_last7",
    "hub_1_lateaircraft_rate_last14",
    "hub_2_depdelay_mean_last1", "hub_2_depdelay_mean_last7",
    "hub_2_depdelay_mean_last14",
    "hub_2_lateaircraft_rate_last1", "hub_2_lateaircraft_rate_last7",
    "hub_2_lateaircraft_rate_last14",
    "hub_3_depdelay_mean_last1", "hub_3_depdelay_mean_last7",
    "hub_3_depdelay_mean_last14",
    "hub_3_lateaircraft_rate_last1", "hub_3_lateaircraft_rate_last7",
    "hub_3_lateaircraft_rate_last14",
    "hub_4_depdelay_mean_last1", "hub_4_depdelay_mean_last7",
    "hub_4_depdelay_mean_last14",
    "hub_4_lateaircraft_rate_last1", "hub_4_lateaircraft_rate_last7",
    "hub_4_lateaircraft_rate_last14",
    "dest_depdelay_mean_last1", "dest_depdelay_mean_last7",
    "dest_depdelay_mean_last14",
    "dest_lateaircraft_rate_last1", "dest_lateaircraft_rate_last7",
    "dest_lateaircraft_rate_last14",
    "origin_congestion_3h_total", "origin_airline_congestion_3h_total",
    "is_holiday",
    "origin_temp_max_K", "origin_temp_min_K", "origin_daily_precip_sum_mm",
    "origin_daily_windspeed_max_kmh", "origin_daily_windgusts_max_kmh",
    "origin_dep_temp_K", "origin_dep_precip_mm", "origin_dep_windspeed_kmh",
    "origin_dep_windgusts_kmh", "origin_dep_visibility_m",
    "origin_dep_cape_jkg", "origin_dep_log1p_cape", "origin_dep_cloudcover_pct",
]
ARR_FEATURE_ORDER = ARR_CAT_FEATURES + ARR_NUM_FEATURES

DEP_TARGETS = ["DepDelayMinutes", "y_dep_ge15", "y_dep_ge30", "y_dep_ge45", "y_dep_ge60"]
ARR_TARGETS = ["ArrDelayMinutes", "y_arr_ge15", "y_arr_ge30", "y_arr_ge45", "y_arr_ge60"]


def load_full_df(airline="WN", target="dep", sample_n=None):
    """Load all v7g model features + targets from processed parquet."""
    if target == "dep":
        cols = DEP_FEATURE_ORDER + DEP_TARGETS + ["FlightDate"]
    else:
        cols = ARR_FEATURE_ORDER + ARR_TARGETS + ["FlightDate"]
    path = DATA_ROOT / f"features_{target}_{airline}_100_v7g_unbalanced.parquet"
    df = pd.read_parquet(path, columns=[c for c in cols if c != "FlightDate"] + ["FlightDate"])
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df


def train_catboost_full(df, feature_order, cat_features, target_col,
                        iterations=1000, depth=6, lr=0.05, verbose=200):
    """Train CatBoost on date-based 80/20 split. Returns model, X_test, y_test, y_prob, auc."""
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import roc_auc_score

    sub = df[feature_order + [target_col, "FlightDate"]].dropna(subset=[target_col])
    for c in cat_features:
        if c in sub.columns:
            sub[c] = sub[c].astype("object").fillna("Unknown").astype(str)

    dates = sorted(sub["FlightDate"].unique())
    split = int(len(dates) * 0.8)
    train = sub[sub["FlightDate"].isin(set(dates[:split]))]
    test = sub[sub["FlightDate"].isin(set(dates[split:]))]

    X_train, y_train = train[feature_order], train[target_col]
    X_test, y_test = test[feature_order], test[target_col]
    cat_idx = [feature_order.index(c) for c in cat_features if c in feature_order]

    model = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=lr,
        loss_function="Logloss", eval_metric="AUC",
        cat_features=cat_idx, verbose=verbose, random_seed=42,
        early_stopping_rounds=50,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train,
              eval_set=Pool(X_test, y_test, cat_features=cat_idx),
              verbose=verbose)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    return model, X_test, y_test, y_prob, auc, cat_idx
