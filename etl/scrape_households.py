from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Iterator
import gzip
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import requests


@dataclass
class HouseholdSource:
    name: str
    source_type: str
    url: Optional[str] = None
    note: Optional[str] = None


DEFAULT_SOURCES: List[HouseholdSource] = [
    HouseholdSource(
        name="ms_idmyph_building_footprints",
        source_type="geojsonl",
        url="https://minedbuildings.z5.web.core.windows.net/legacy/southeast-asia/philippines.geojsonl.zip",
        note=(
            "Philippines-only building footprints (GeoJSONL). Large download; cached if cache_dir is set."
        ),
    ),
]


def get_default_household_dir() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "households"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _timestamped_filename(now=None):
    now = now or pd.Timestamp.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_philippines_buildings_history.csv"


def _safe_get(url: str, timeout: int = 30) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"Warning: request failed for {url}: {exc}")
        return None


def _download_to_cache(url: str, cache_dir: Path) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    out_path = cache_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return out_path
    except Exception as exc:
        print(f"Warning: failed to download {url}: {exc}")
        return None


def _ensure_path(value: str | Path, cache_dir: Optional[Path]) -> Optional[Path]:
    path = Path(value)
    if path.exists():
        return path
    if cache_dir is None:
        return None
    if str(value).startswith("http"):
        return _download_to_cache(str(value), cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / path.name


def _extract_if_needed(path: Path, cache_dir: Path) -> Optional[Path]:
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                if not members:
                    return None
                target = cache_dir / members[0]
                if not target.exists():
                    zf.extract(members[0], cache_dir)
                return target
        except zipfile.BadZipFile:
            print(f"Warning: failed to extract {path}: bad zip file")
            return None
        except Exception as exc:
            print(f"Warning: failed to extract {path}: {exc}")
            return None
    if path.suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".gz":
        try:
            if path.suffixes[-2:] == [".csv", ".gz"]:
                target = cache_dir / f"{path.stem}.geojsonl"
            else:
                target = cache_dir / path.with_suffix("").name
            if not target.exists():
                with gzip.open(path, "rb") as f_in:
                    with open(target, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            return target
        except Exception as exc:
            print(f"Warning: failed to gunzip {path}: {exc}")
            return None
    return path


def _load_google_open_buildings_csv(path_or_url: str, bbox: Optional[Tuple[float, float, float, float]]):
    if path_or_url.startswith("http"):
        data = _safe_get(path_or_url)
        if data is None:
            return pd.DataFrame()
        df = pd.read_csv(BytesIO(data))
    else:
        df = pd.read_csv(path_or_url)

    if df.empty:
        return df

    rename_map = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "area_in_meters": "House_Surface_Area_Sqm",
        "area_in_meters_sq": "House_Surface_Area_Sqm",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    if bbox and "Latitude" in df.columns and "Longitude" in df.columns:
        min_lon, min_lat, max_lon, max_lat = bbox
        df = df[
            (df["Longitude"].between(min_lon, max_lon))
            & (df["Latitude"].between(min_lat, max_lat))
        ]

        return df[[c for c in ["Latitude", "Longitude", "House_Surface_Area_Sqm"] if c in df.columns]]


def _load_geojsonl_with_geopandas(
    path_or_url: str,
    bbox: Optional[Tuple[float, float, float, float]],
    cache_dir: Optional[Path] = None,
    max_rows: Optional[int] = None,
):
    try:
        import geopandas as gpd
        from shapely.geometry import shape as shapely_shape
    except Exception:
        print("Warning: geopandas is not installed. Falling back to manual GeoJSON parsing.")
        return _load_geojsonl_manual(path_or_url, bbox, cache_dir, max_rows=max_rows)

    # --- resolve file path (download / extract if needed) ---
    resolved: Optional[Path] = None
    if path_or_url.startswith("http"):
        if cache_dir is None:
            data = _safe_get(path_or_url)
            if data is None:
                return pd.DataFrame()
            resolved = BytesIO(data)  # type: ignore[assignment]
        else:
            downloaded = _download_to_cache(path_or_url, cache_dir)
            if downloaded is None:
                return pd.DataFrame()
            extracted = _extract_if_needed(downloaded, cache_dir)
            if extracted is None and downloaded.suffix == ".zip":
                try:
                    downloaded.unlink(missing_ok=True)
                except Exception:
                    pass
                downloaded = _download_to_cache(path_or_url, cache_dir)
                if downloaded is None:
                    return pd.DataFrame()
                extracted = _extract_if_needed(downloaded, cache_dir)
            if extracted is None:
                return pd.DataFrame()
            resolved = extracted
    else:
        path = Path(path_or_url)
        if cache_dir is not None:
            extracted = _extract_if_needed(path, cache_dir)
            path = extracted if extracted is not None else path
        resolved = path

    # --- Stream with Fiona for bbox + max_rows (avoids loading entire file) ---
    try:
        import fiona
        resolved_str = str(resolved) if not isinstance(resolved, BytesIO) else None
        if resolved_str is not None:
            geometries = []
            with fiona.open(resolved_str) as src:
                iterator = src.filter(bbox=bbox) if bbox else iter(src)
                for feat in iterator:
                    geom = shapely_shape(feat["geometry"])
                    if geom is not None and not geom.is_empty:
                        geometries.append(geom)
                    if max_rows is not None and len(geometries) >= max_rows:
                        break
            if not geometries:
                print(f"Warning: no features found in bbox {bbox}")
                return pd.DataFrame()
            gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        else:
            # BytesIO fallback — small in-memory data, just read all
            gdf = gpd.read_file(resolved)
            if bbox and not gdf.empty:
                min_lon, min_lat, max_lon, max_lat = bbox
                gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]
            if max_rows is not None and len(gdf) > max_rows:
                gdf = gdf.sample(n=max_rows, random_state=42)
    except Exception as exc:
        print(f"Warning: Fiona streaming failed ({exc}); falling back to manual parser.")
        return _load_geojsonl_manual(path_or_url, bbox, cache_dir, max_rows=max_rows)

    if gdf.empty:
        return pd.DataFrame()

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf_proj = gdf.to_crs("EPSG:3857")

    gdf["House_Surface_Area_Sqm"] = gdf_proj.geometry.area
    centroids_proj = gdf_proj.geometry.centroid
    centroids = centroids_proj.to_crs("EPSG:4326")
    gdf["Latitude"] = centroids.y
    gdf["Longitude"] = centroids.x

    return gdf.reindex(columns=["Latitude", "Longitude", "House_Surface_Area_Sqm"])


def _iter_features_from_geojson(path: Path) -> Iterator[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "{":
                payload = json.load(f)
                if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
                    for feature in payload.get("features", []):
                        if isinstance(feature, dict):
                            yield feature
                elif isinstance(payload, dict) and payload.get("type") == "Feature":
                    yield payload
                else:
                    return
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        feature = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(feature, dict):
                        yield feature
    except Exception as exc:
        print(f"Warning: failed to read GeoJSON/GeoJSONL: {exc}")


def _polygon_centroid_area(coords: List[List[float]]):
    if not coords:
        return None, None
    ring = coords[0]
    if len(ring) < 3:
        return None, None
    x = [p[0] for p in ring]
    y = [p[1] for p in ring]
    n = len(ring)
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n - 1):
        cross = x[i] * y[i + 1] - x[i + 1] * y[i]
        area2 += cross
        cx += (x[i] + x[i + 1]) * cross
        cy += (y[i] + y[i + 1]) * cross
    if area2 == 0:
        return None, None
    area = area2 / 2.0
    cx = cx / (3.0 * area2)
    cy = cy / (3.0 * area2)
    return (cx, cy), abs(area)


def _area_deg2_to_m2(area_deg2: float, lat: float) -> float:
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat))
    return area_deg2 * m_per_deg_lon * m_per_deg_lat


def _load_geojsonl_manual(path_or_url: str, bbox: Optional[Tuple[float, float, float, float]], cache_dir: Optional[Path] = None, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path_or_url)
    if path_or_url.startswith("http"):
        if cache_dir is None:
            print("Warning: manual GeoJSON parsing requires cache_dir for download.")
            return pd.DataFrame()
        downloaded = _download_to_cache(path_or_url, cache_dir)
        if downloaded is None:
            return pd.DataFrame()
        extracted = _extract_if_needed(downloaded, cache_dir)
        if extracted is None and downloaded.suffix == ".zip":
            try:
                downloaded.unlink(missing_ok=True)
            except Exception:
                pass
            downloaded = _download_to_cache(path_or_url, cache_dir)
            if downloaded is None:
                return pd.DataFrame()
            extracted = _extract_if_needed(downloaded, cache_dir)
        if extracted is None:
            return pd.DataFrame()
        path = extracted
    else:
        if cache_dir is not None:
            extracted = _extract_if_needed(path, cache_dir)
            path = extracted if extracted is not None else path

    rows = []
    for feature in _iter_features_from_geojson(path):
        geom = feature.get("geometry") if isinstance(feature, dict) else None
        if not geom:
            continue
        geom_type = geom.get("type")
        coords = geom.get("coordinates")
        if geom_type != "Polygon" or not coords:
            continue

        centroid, area_deg2 = _polygon_centroid_area(coords)
        if centroid is None or area_deg2 is None:
            continue
        lon, lat = centroid

        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                continue

        area_m2 = _area_deg2_to_m2(area_deg2, lat)
        props = feature.get("properties", {}) or {}
        height = None
        if "height" in props:
            height = props.get("height")
        elif "building:levels" in props:
            try:
                height = float(props.get("building:levels")) * 3
            except Exception:
                height = None

        rows.append(
            {
                "Latitude": lat,
                "Longitude": lon,
                "House_Surface_Area_Sqm": area_m2,
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _load_osm_pbf_with_pyrosm(path_or_url: str, bbox: Optional[Tuple[float, float, float, float]], cache_dir: Optional[Path] = None):
    try:
        from pyrosm import OSM
    except Exception:
        print("Warning: pyrosm is not installed. Skipping OSM PBF source.")
        return pd.DataFrame()

    path = path_or_url
    if path_or_url.startswith("http"):
        if cache_dir is None:
            print("Warning: OSM PBF requires cache_dir for download. Provide cache_dir or local file path.")
            return pd.DataFrame()
        downloaded = _download_to_cache(path_or_url, cache_dir)
        if downloaded is None:
            return pd.DataFrame()
        path = str(downloaded)

    osm = OSM(path)
    buildings = osm.get_buildings()
    if buildings is None or buildings.empty:
        return pd.DataFrame()

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        buildings = buildings.cx[min_lon:max_lon, min_lat:max_lat]

    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings["House_Surface_Area_Sqm"] = buildings.geometry.to_crs("EPSG:3857").area
    buildings["Latitude"] = buildings.geometry.centroid.y
    buildings["Longitude"] = buildings.geometry.centroid.x

    if "height" in buildings.columns:
        buildings["House_Height_Meters"] = pd.to_numeric(buildings["height"], errors="coerce")
    elif "building:levels" in buildings.columns:
        buildings["House_Height_Meters"] = pd.to_numeric(buildings["building:levels"], errors="coerce") * 3

    return buildings[["Latitude", "Longitude", "House_Surface_Area_Sqm", "House_Height_Meters"]]


def _load_open_buildings_tiles(url: str, bbox: Optional[Tuple[float, float, float, float]], cache_dir: Optional[Path]):
    try:
        import geopandas as gpd
    except Exception:
        print("Warning: geopandas is not installed. Skipping Open Buildings tiles.")
        return pd.DataFrame()

    data = _safe_get(url)
    if data is None:
        return pd.DataFrame()
    tiles = gpd.read_file(BytesIO(data))
    if tiles.empty:
        return pd.DataFrame()

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        tiles = tiles.cx[min_lon:max_lon, min_lat:max_lat]

    if tiles.empty:
        return pd.DataFrame()

    url_keys = [c for c in tiles.columns if "url" in c.lower()]
    if not url_keys:
        print("Warning: Open Buildings tiles metadata has no URL column.")
        return pd.DataFrame()

    tile_urls = tiles[url_keys[0]].dropna().astype(str).unique().tolist()
    if not tile_urls:
        return pd.DataFrame()

    frames = []
    for tile_url in tile_urls[:5]:
        df = _load_google_open_buildings_csv(tile_url, bbox)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_ms_global_dataset_links(url: str, bbox: Optional[Tuple[float, float, float, float]], cache_dir: Optional[Path]):
    data = _safe_get(url)
    if data is None:
        return pd.DataFrame()

    try:
        df_links = pd.read_csv(BytesIO(data))
    except Exception as exc:
        print(f"Warning: failed to parse dataset-links CSV: {exc}")
        return pd.DataFrame()

    if df_links.empty:
        return df_links

    cols = {c.lower(): c for c in df_links.columns}
    country_col = cols.get("country") or cols.get("country_name") or cols.get("region")
    url_col = cols.get("url") or cols.get("link") or cols.get("download_url")

    if not url_col:
        print("Warning: dataset-links CSV missing URL column.")
        return pd.DataFrame()

    if country_col:
        df_links = df_links[df_links[country_col].astype(str).str.contains("philippines", case=False, na=False)]

    if df_links.empty:
        print("Warning: no Philippines entries found in dataset-links CSV.")
        return pd.DataFrame()

    frames = []
    for link in df_links[url_col].dropna().astype(str).head(5).tolist():
        df = _load_geojsonl_with_geopandas(link, bbox, cache_dir)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_household_measurements(
    sources: Iterable[HouseholdSource] = DEFAULT_SOURCES,
    *,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    cache_dir: Optional[Path] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for source in sources:
        if not source.url:
            print(f"Warning: {source.name} has no URL/path. {source.note or ''}")
            continue

        path_or_url = source.url
        if cache_dir is not None and path_or_url.startswith("http"):
            cached_path = _ensure_path(path_or_url, cache_dir)
            if cached_path and cached_path.exists():
                path_or_url = str(cached_path)

        if source.source_type == "geojsonl":
            df = _load_geojsonl_with_geopandas(path_or_url, bbox, cache_dir, max_rows=max_rows)
        else:
            print(f"Warning: Unsupported source type {source.source_type} for {source.name}.")
            df = pd.DataFrame()

        if df is not None and not df.empty:
            df = df.copy()
            df["Source"] = source.name
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "House_Height_Meters" in combined.columns:
        combined = combined.drop(columns=["House_Height_Meters"])
    return combined


def refresh_household_cache(
    sources: Iterable[HouseholdSource] = DEFAULT_SOURCES,
    *,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    out_dir = cache_dir or get_default_household_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_household_measurements(sources, bbox=bbox, cache_dir=out_dir)
    out_path = out_dir / _timestamped_filename()
    df.to_csv(out_path, index=False)
    return out_path


def generate_base_training_data(
    df_hist_agg: pd.DataFrame,
    num_households: int,
    *,
    household_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df_hist_sample = df_hist_agg.reset_index(drop=True)

    household_df = household_df.copy() if household_df is not None else pd.DataFrame()
    required_cols = ["Latitude", "Longitude", "House_Surface_Area_Sqm"]
    missing_cols = [col for col in required_cols if col not in household_df.columns]
    if missing_cols:
        raise ValueError(
            "Household data is missing required columns: "
            + ", ".join(missing_cols)
        )

    household_sample = household_df.dropna(subset=required_cols).reset_index(drop=True)
    if household_sample.empty:
        raise ValueError("Household data has no valid rows for required columns.")

    n_flood = len(df_hist_sample)
    n_house = len(household_sample)

    if num_households is None:
        num_rows = n_house  # use all available households
    else:
        num_rows = int(num_households)

    # Sample or tile households to reach num_rows
    if n_house >= num_rows:
        household_sample = household_sample.sample(n=num_rows, random_state=42).reset_index(drop=True)
    else:
        reps = (num_rows // n_house) + 1
        household_sample = pd.concat(
            [household_sample] * reps, ignore_index=True
        ).iloc[:num_rows].reset_index(drop=True)

    # Tile flood locations to match num_rows (cycle through all locations)
    if n_flood < num_rows:
        reps = (num_rows // n_flood) + 1
        df_hist_sample = pd.concat(
            [df_hist_sample] * reps, ignore_index=True
        ).iloc[:num_rows].reset_index(drop=True)
    else:
        df_hist_sample = df_hist_sample.iloc[:num_rows].reset_index(drop=True)

    area_values = pd.to_numeric(
        household_sample["House_Surface_Area_Sqm"], errors="coerce"
    ).fillna(0)
    height_values = np.sqrt(area_values)

    flood_depth_values = df_hist_sample["flood_depth_m"].fillna(0).values
    safe_height = np.where(height_values.to_numpy() == 0, np.nan, height_values.to_numpy())
    flood_ratio = flood_depth_values / safe_height
    damage_classification = np.select(
        [flood_ratio >= 0.8, flood_ratio >= 0.4],
        ["Totally Damaged", "Partially Damaged"],
        default="Not Damaged",
    )

    data = {
        "Location": df_hist_sample["Location"].values,
        "Latitude": household_sample["Latitude"].values,
        "Longitude": household_sample["Longitude"].values,
        "Flood_Depth_Meters": flood_depth_values,
        "House_Surface_Area_Sqm": area_values.values,
        "Sensor_Flood_Depth_Meters": df_hist_sample["sensor_flood_depth_m"].fillna(0).values,
        "Damage_Classification": damage_classification,
    }
    df = pd.DataFrame(data)
    if list(df.columns) != list(data.keys()):
        raise ValueError("Generated data columns do not match data dictionary keys.")
    return df


if __name__ == "__main__":
    output_path = refresh_household_cache()
    print(f"Saved: {output_path}")
