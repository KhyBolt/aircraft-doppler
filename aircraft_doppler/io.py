"""Explicit schemas rather than implicit provider units or altitude guesses."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .physics import Track, geodetic_to_ecef
from .search import Event, SearchSettings

XYZ = {"x_m", "y_m", "z_m"}
GEOGRAPHIC = {"latitude", "longitude", "altitude_ellipsoid_m"}


def load_table(path):
    return pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)


def geographic_to_cartesian(frame):
    required = {"latitude", "longitude", "altitude_ellipsoid_m"}
    if not required.issubset(frame.columns):
        raise ValueError("Geographic input needs latitude, longitude, altitude_ellipsoid_m; convert other height datums first")
    xyz = geodetic_to_ecef(frame.latitude.to_numpy(), frame.longitude.to_numpy(), frame.altitude_ellipsoid_m.to_numpy())
    frame = frame.copy()
    frame[["x_m", "y_m", "z_m"]] = xyz
    return frame


def _normalize_frame(data):
    columns = set(data.columns)
    if GEOGRAPHIC & columns:
        converted = geographic_to_cartesian(data)
        xyz = ["x_m", "y_m", "z_m"]
        if (XYZ & columns and (not XYZ <= columns
                or not np.allclose(data[xyz].to_numpy(float), converted[xyz].to_numpy(float),
                                   atol=1e-3, rtol=0))):
            raise ValueError("Geographic and Cartesian columns disagree with WGS84 ECEF")
        return converted, "ecef"
    if not XYZ <= columns:
        raise ValueError("Need complete x_m/y_m/z_m or latitude/longitude/altitude_ellipsoid_m")
    return data, "cartesian"


def load_tracks(path, max_gap_s=15, *, return_frame=False):
    data = (pd.read_csv(path, dtype={"flight_id": "string", "aircraft_id": "string"})
            if Path(path).suffix.lower() != ".parquet" else load_table(path))
    data, frame = _normalize_frame(data)
    required = {"flight_id", "time_s", "x_m", "y_m", "z_m"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing track columns {required-set(data.columns)}")
    if data.empty or data["flight_id"].isna().any():
        raise ValueError("Track table must contain nonempty flight IDs")
    data["flight_id"] = data["flight_id"].astype(str).str.strip()
    if data.flight_id.eq("").any() or data.flight_id.str.contains("|", regex=False).any():
        raise ValueError("Empty flight IDs and reserved separator | are not allowed")
    tracks = []
    for fid, group in data.groupby("flight_id", sort=True):
        group = group.sort_values("time_s")
        if group.time_s.duplicated().any():
            raise ValueError(f"Duplicate timestamps in {fid}; resolve them at ingestion")
        engine = "unknown"
        if "engine" in group:
            labels = set(group.engine.dropna().astype(str).str.lower().str.strip()) - {"", "unknown"}
            if len(labels) > 1 or not labels <= {"propeller", "jet"}:
                raise ValueError(f"Inconsistent or unrecognized engine labels in {fid}")
            engine = next(iter(labels), "unknown")
        aircraft_id = None
        if "aircraft_id" in group:
            ids = set(group.aircraft_id.dropna().astype(str).str.strip()) - {""}
            if len(ids) > 1:
                raise ValueError(f"Inconsistent aircraft_id in {fid}")
            aircraft_id = next(iter(ids), None)
        tracks.append(Track(str(fid), group.time_s.to_numpy(), group[["x_m", "y_m", "z_m"]].to_numpy(),
                            engine, max_gap_s, aircraft_id))
    return (tracks, frame) if return_frame else tracks


def load_candidates(path, *, return_frame=False):
    data, frame = _normalize_frame(load_table(path))
    xyz = ["x_m", "y_m", "z_m"]
    data[xyz] = data[xyz].apply(pd.to_numeric, errors="raise")
    if not len(data) or not np.all(np.isfinite(data[xyz].to_numpy())) or data[xyz].duplicated().any():
        raise ValueError("Candidate positions must be nonempty, unique, and finite")
    data = data.reset_index(drop=True)
    return (data, frame) if return_frame else data


def load_inputs(track_path, candidate_path):
    tracks, track_frame = load_tracks(track_path, return_frame=True)
    candidates, candidate_frame = load_candidates(candidate_path, return_frame=True)
    if track_frame != candidate_frame:
        raise ValueError("Track and candidate coordinate frames disagree: use both geographic or both Cartesian")
    return tracks, candidates, track_frame


def load_config(path):
    with open(path) as f:
        obj = json.load(f)
    return [Event(**x) for x in obj["events"]], SearchSettings(**obj.get("search", {})), obj.get("stft", {})


def write_geographic_grid(bounds, spacing_m, dem_path, geoid_path, output, microphone_height_m=1.5):
    """Sample an orthometric DEM + geoid undulation raster into ECEF.

    Requires optional rasterio + pyproj. Both rasters may have any rasterio CRS.
    h_ellipsoid = H_orthometric + N_geoid + microphone_height.
    The DEM and geoid MUST have compatible stated vertical datums and metres.
    No automatic inference of their vertical metadata is possible here.
    Bounds: [west, south, east, north], within one local UTM zone.
    """
    import rasterio
    from pyproj import Transformer
    west, south, east, north = bounds
    if not -180 <= west < east <= 180 or not -80 <= south < north <= 84 or spacing_m <= 0:
        raise ValueError("Invalid local non-antimeridian bounds or spacing")
    zone = int(((west+east)/2+180)//6)+1
    west_zone = min(60, int((west+180)//6)+1)
    east_zone = min(60, int((np.nextafter(east,west)+180)//6)+1)
    if west_zone != east_zone or south < 0 < north:
        raise ValueError("Use local tiles within one hemisphere and UTM zone")
    epsg = (32600 if (south+north)/2 >= 0 else 32700)+zone
    to_local = Transformer.from_crs(4326, epsg, always_xy=True)
    to_geo = Transformer.from_crs(epsg, 4326, always_xy=True)
    bx, by = to_local.transform([west,west,east,east], [south,north,south,north])
    xx, yy = np.meshgrid(np.arange(min(bx), max(bx)+spacing_m, spacing_m), np.arange(min(by), max(by)+spacing_m, spacing_m))
    lon, lat = to_geo.transform(xx.ravel(), yy.ravel())
    keep = (lon >= west)&(lon <= east)&(lat >= south)&(lat <= north)
    lon, lat = lon[keep], lat[keep]
    def sample(path):
        with rasterio.open(path) as raster:
            if raster.crs is None or raster.nodata is None:
                raise ValueError("Raster must declare CRS and nodata")
            transformer = Transformer.from_crs(4326, raster.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            values = np.ma.vstack(list(raster.sample(zip(x,y), masked=True))).astype(float)[:,0]
            return np.asarray(values.filled(np.nan), float)
    height = sample(dem_path)+sample(geoid_path)+microphone_height_m
    result = pd.DataFrame({"latitude": lat, "longitude": lon, "altitude_ellipsoid_m": height})
    result = result[np.isfinite(height)]
    if not len(result):
        raise ValueError("No valid elevations in bounds")
    geographic_to_cartesian(result).to_csv(output, index=False)
    return len(result)
