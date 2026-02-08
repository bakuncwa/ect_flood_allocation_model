from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL.*",
)

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

DEFAULT_ENDPOINTS = {
    "dost_asti_sensors": "https://philsensors.asti.dost.gov.ph/site/waterlevel",
}


def get_default_cache_path():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "waterlevel"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "philippines_waterlevel_history.csv"


def _timestamped_filename(now=None):
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_philippines_waterlevel_history.csv"


def refresh_waterlevel_cache(*, cache_dir: Path, endpoints: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = run_dost_etl(endpoints)
    filename = _timestamped_filename()
    out_path = cache_dir / filename
    df_to_cache = df.copy()
    source = df.attrs.get("source")
    if source:
        df_to_cache["__source"] = source
    df_to_cache.to_csv(out_path, index=False)
    return df, out_path


def refresh_realtime_waterlevel(*, endpoints: dict = DEFAULT_ENDPOINTS):
    cache_path = get_default_cache_path()
    df, out_path = refresh_waterlevel_cache(cache_dir=cache_path.parent, endpoints=endpoints)
    return df, out_path


def _parse_meter_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("m", "").replace("M", "")
    text = "".join(ch for ch in text if ch.isdigit() or ch in {".", "-"})
    try:
        return float(text) if text else None
    except ValueError:
        return None


def _fetch_monitoring_payload(base_url):
    page_url = f"{base_url}/site/waterlevel"
    post_url = f"{base_url}/station/monitoring"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    session = requests.Session()
    page_resp = session.get(page_url, headers=headers, timeout=20)
    page_resp.raise_for_status()
    soup = BeautifulSoup(page_resp.text, "html.parser")
    csrf = soup.find("meta", {"name": "csrf-token"})
    csrf_token = csrf["content"] if csrf else None

    post_headers = {
        **headers,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }
    if csrf_token:
        post_headers["X-CSRF-Token"] = csrf_token

    post_resp = session.post(post_url, headers=post_headers, timeout=20)
    post_resp.raise_for_status()
    return post_resp.json()


def _scrape_waterlevel_latest(url):
    base_url = "https://philsensors.asti.dost.gov.ph"
    try:
        payload = _fetch_monitoring_payload(base_url)
    except Exception as exc:
        print(f"Warning: monitoring endpoint failed: {exc}")
        payload = None

    if payload and isinstance(payload, dict):
        water_rows = payload.get("data_water", [])
        if water_rows:
            df = pd.DataFrame(water_rows)
            df = df.rename(
                columns={
                    "region": "Region",
                    "province": "Province",
                    "location": "Location",
                }
            )

            hour_keys = [c for c in df.columns if str(c).isdigit()]
            hour_keys = sorted(hour_keys, key=lambda x: int(x))
            now = datetime.now()
            hour_labels = []
            for i, key in enumerate(hour_keys):
                label = (now - timedelta(hours=int(key))).strftime("%H:00")
                if label in hour_labels:
                    label = f"{label} (prev)"
                hour_labels.append(label)

            for key, label in zip(hour_keys, hour_labels):
                df[label] = df[key].map(_parse_meter_value)

            if "0" in df.columns:
                df["Current Hour"] = df["0"].map(_parse_meter_value)
            else:
                df["Current Hour"] = pd.Series([np.nan] * len(df))

            df["date"] = datetime.today().strftime("%Y-%m-%d")
            df["Region"] = df["Region"].fillna("").astype(str).str.strip()
            df["Province"] = df["Province"].fillna("").astype(str).str.strip()
            df["Location"] = df["Location"].fillna("").astype(str).str.strip()

            df["sensor_flood_depth_m"] = df["Current Hour"]
            df = df.dropna(subset=["Location", "sensor_flood_depth_m"])
            return df

    return pd.DataFrame()


def run_dost_etl(endpoints):
    """
    ETL pipeline for DOST-ASTI waterlevel monitoring (Philippines-wide).
    """
    def _synthetic_fallback(reason):
        raise ValueError(
            "Refusing synthetic fallback. "
            f"{reason}. Provide official URLs/structured datasets that include Location."
        )

    sensors_url = endpoints.get("dost_asti_sensors")
    if not sensors_url:
        return _synthetic_fallback("Missing DOST-ASTI sensors endpoint")

    sensors_df = (
        _scrape_waterlevel_latest(sensors_url)
        if sensors_url
        else pd.DataFrame()
    )

    if sensors_df.empty:
        return _synthetic_fallback("DOST-ASTI sensors table not found or empty")

    df = sensors_df.copy()
    df.attrs["source"] = "etl"

    if "flood_depth_m" not in df.columns and "sensor_flood_depth_m" in df.columns:
        df["flood_depth_m"] = df["sensor_flood_depth_m"].clip(0, 5)

    required_cols = {"date", "Location"}
    if not required_cols.issubset(set(df.columns)):
        return _synthetic_fallback("Scraped tables missing required columns: date/Location")
    return df


def load_historical_flood_data(*, cache_path, endpoints):
    cache_path = Path(cache_path)
    cache_dir = cache_path.parent
    df, _ = refresh_waterlevel_cache(cache_dir=cache_dir, endpoints=endpoints)
    return df


def load_and_aggregate_flood_data(*, cache_path, endpoints):
    df_hist = load_historical_flood_data(
        cache_path=cache_path,
        endpoints=endpoints,
    )
    df_hist["date"] = pd.to_datetime(df_hist.get("date"), errors="coerce")
    if "Location" not in df_hist.columns:
        raise ValueError("Historical flood dataset missing 'Location' column. Check ETL sources.")
    df_hist = df_hist.dropna(subset=["Location"])

    if df_hist.empty:
        raise ValueError("Historical flood dataset is empty. Check the ETL cache or DOST-ASTI source.")

    df_hist_agg = (
        df_hist.groupby(["Location", "Province", "Region"], as_index=False)
        .agg(
            {
                "sensor_flood_depth_m": "mean",
                "flood_depth_m": "mean",
            }
        )
    )
    return df_hist, df_hist_agg


if __name__ == "__main__":
    df_latest, output_path = refresh_realtime_waterlevel()
    print(f"Saved: {output_path}")
    print(df_latest.head())
