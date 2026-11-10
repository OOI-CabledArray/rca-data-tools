import ast
import io
import numpy as np
import os
import pandas as pd
import re
import requests
import s3fs
import xarray as xr
import importlib

def select_logger():
    from prefect import get_run_logger
    try:
        logger = get_run_logger()
    except:
        from loguru import logger
    
    return logger


QARTOD_NOT_EVALUATED = 2


def _qartod_digit(a, i, n, var, test, logger):
    """i-th character of each qartod_executed string as an int.

    Reads the fixed-width unicode codepoints directly instead of a python call
    per element (~250x faster than `.str[i].astype(int)` on a 12M-point chunk).
    A character that is not a digit (e.g. 'n' from a 'nan' fill squeezed into
    a one-character string, or '' when the flag is missing) has no test result,
    so it becomes 2 (not evaluated) and is reported once per chunk.
    """
    u = np.asarray(a).astype(f"U{n}")
    d = u.view(np.uint32).reshape(u.shape + (n,))[..., i].astype(int) - ord("0")
    invalid = (d < 0) | (d > 9)
    if invalid.any():
        values = sorted({chr(c + ord("0")) if c + ord("0") else "" for c in np.unique(d[invalid])})
        logger.warning(
            f"Invalid QARTOD flag value(s) {values} in {var} ({test}): "
            f"{invalid.sum()} point(s) plotted as {QARTOD_NOT_EVALUATED} (not evaluated)"
        )
        d = np.where(invalid, QARTOD_NOT_EVALUATED, d)
    return d


def coerce_qartod_executed_to_int(ds):
    logger = select_logger()

    qartod_executed_vars = [var for var in ds.variables if 'qartod_executed' in var]
    for var in qartod_executed_vars:
        executed_tests = ds[var].tests_executed.replace(' ', '').split(',')

        for i, test in enumerate(executed_tests):
            test_var_name = f"{var}_{test}"
            # the decode runs lazily in dask threads, which have no Prefect run
            # context; pass this logger in so warnings reach the flow-run logs
            ds[test_var_name] = xr.apply_ufunc(
                _qartod_digit, ds[var],
                kwargs={"i": i, "n": len(executed_tests), "var": var, "test": test, "logger": logger},
                dask="parallelized", output_dtypes=[int],
            )

        ds = ds.drop(var)
    logger.info(f"ds size post coercion: {ds.nbytes}")
    return ds


def get_s3_kwargs():
    aws_key = os.environ.get("AWS_KEY")
    aws_secret = os.environ.get("AWS_SECRET")
    
    s3_kwargs = {'key': aws_key, 'secret': aws_secret}
    return s3_kwargs


def open_store(zarr_file, bucket="ooi-data"):
    """Lazy dataset for a zarr store, in ooi-data unless bucket is given."""
    fs = s3fs.S3FileSystem(**get_s3_kwargs())
    return xr.open_zarr(fs.get_mapper(f"{bucket}/{zarr_file}"), consolidated=True)


PROFILE_INDEX_URL = "https://raw.githubusercontent.com/OOI-CabledArray/profileIndices/main/"


def load_profiles(refDes, years=None):
    """RCA profiler indices (profile, start, peak, end) for refDes's site, sorted by start: one
    CSV per year, 2014 to now unless years is given. [] if there are none."""
    site = refDes.split("-")[0]
    parts = []
    for year in years or range(2014, pd.Timestamp.now().year + 1):
        download = requests.get(f"{PROFILE_INDEX_URL}{site}_profiles_{year}.csv",
                                headers={"User-Agent": "RCA-profile-fetcher"})
        if download.status_code == 200:
            parts.append(pd.read_csv(io.StringIO(download.text), parse_dates=["start", "peak", "end"]))
    return pd.concat(parts, ignore_index=True).sort_values("start") if parts else []


def ecs_job_variables(size):
    """Prefect job_variables for an ECS size string like "16vcpu_88gb"."""
    vcpu, gb = size.removesuffix("gb").split("vcpu_")
    return {"cpu": int(vcpu) * 1024, "memory": int(gb) * 1024}


def save_fig(fig, fileNameList, fileName, dpi, imgTagList):
    for imgTag in imgTagList:
        imgTag = imgTag + '.png'
        fileNameList.append(fileName + imgTag)
        fig.savefig(fileName + imgTag, dpi=dpi)
        
        
# --------------------------------
# OPTAA calibration loadhelpers
# --------------------------------

def _load_ext_array(url):
    """
    Fetch a .ext file and parse it as a 2D float array.

    Each line is a comma-separated row; returns shape (n_rows, n_cols).
    """
    resp = requests.get(url)
    resp.raise_for_status()
    rows = []
    for line in resp.text.strip().splitlines():
        rows.append([float(v) for v in line.split(",")])
    return np.array(rows, dtype=float)


def _resolve_sheetrefs(coef_dict, csv_url):
    """
    Replace any SheetRef: values in coef_dict with the actual 2D array
    loaded from the corresponding .ext file.

    The .ext file URL is derived from the CSV URL by replacing
    '.csv' with '__{sheet_name}.ext', matching the OOI asset-management
    naming convention.

    Parameters
    ----------
    coef_dict : dict
        Coefficient name -> value, as read from the calibration CSV.
    csv_url : str
        Full URL of the source CSV file.

    Returns
    -------
    dict
        coef_dict with SheetRef values replaced by 2D numpy arrays.
    """
    resolved = {}
    for name, value in coef_dict.items():
        if isinstance(value, str) and value.startswith("SheetRef:"):
            sheet_name = value.split(":", 1)[1].strip()
            ext_url = csv_url.replace(".csv", f"__{sheet_name}.ext")
            resolved[name] = _load_ext_array(ext_url)
        else:
            resolved[name] = value
    return resolved


# -----------------------------------------------
# Main calibration coefficient load functions
# -----------------------------------------------

def get_calibration_dataset(reference_designator):
    """
    Load full calibration history for a reference designator directly from GitHub,
    including only calibration events that overlap actual deployments.

    Parameters
    ----------
    reference_designator : str
        e.g. "CE04OSPS-SF01B-4A-NUTNRA102"

    Returns
    -------
    xr.Dataset
        Dataset of calibration coefficients with dimensions:
        - calibration (time of calibration)
        - {coef}_index (for 1-D array coefficients)
        - {coef}_row / {coef}_col (for 2-D array coefficients, e.g. OPTAA)
    """
    GITHUB_API = "https://api.github.com/repos/oceanobservatories/asset-management/contents"
    RAW_BASE = "https://raw.githubusercontent.com/oceanobservatories/asset-management/master"

    # ---------------------------
    # 1. Load deployment file
    # ---------------------------
    deploy_url = f"{RAW_BASE}/deployment/{reference_designator[:8]}_Deploy.csv"
    df_deploy = pd.read_csv(deploy_url)

    df_deploy = df_deploy.rename(columns={
        "Reference Designator": "referenceDesignator",
        "sensor.uid": "assetUid"
    })
    df_deploy = df_deploy[df_deploy["referenceDesignator"] == reference_designator].copy()
    if df_deploy.empty:
        raise ValueError(f"No deployments found for {reference_designator}")

    df_deploy["start"] = pd.to_datetime(df_deploy["startDateTime"], errors="coerce")
    df_deploy["stop"] = pd.to_datetime(df_deploy["stopDateTime"], errors="coerce").fillna(pd.Timestamp.max)
    df_deploy = df_deploy.dropna(subset=["start"]).sort_values("start")

    # ---------------------------
    # 2. List all calibration files for this instrument class
    # ---------------------------
    instrument_class = reference_designator.split("-")[-1][:6]
    calib_api_url = f"{GITHUB_API}/calibration/{instrument_class}"
    resp = requests.get(calib_api_url)
    resp.raise_for_status()
    calib_files = resp.json()

    records = []
    for f in calib_files:
        name = f["name"]
        match = re.match(r"(.*)__(\d{8})\.csv", name)
        if match:
            records.append({
                "asset_uid": match.group(1),
                "cal_date": pd.to_datetime(match.group(2), format="%Y%m%d"),
                "download_url": f["download_url"]
            })
    df_cal_index = pd.DataFrame(records)
    if df_cal_index.empty:
        raise ValueError("No calibration files found")

    deployed_uids = df_deploy["assetUid"].unique()
    df_cal_index = df_cal_index[df_cal_index["asset_uid"].isin(deployed_uids)].sort_values(["asset_uid", "cal_date"])
    if df_cal_index.empty:
        raise ValueError("No calibration files match deployed assets")

    # ---------------------------
    # 3. Deployment-aware calibration filtering
    # ---------------------------
    coef_records = []

    for dep in df_deploy.itertuples():
        asset_uid = dep.assetUid
        dep_start = dep.start
        dep_stop = dep.stop

        df_cal_uid = df_cal_index[df_cal_index["asset_uid"] == asset_uid].sort_values("cal_date")
        if df_cal_uid.empty:
            continue

        df_cal_uid = df_cal_uid.copy()
        df_cal_uid["valid_start"] = df_cal_uid["cal_date"]
        df_cal_uid["valid_stop"] = df_cal_uid["cal_date"].shift(-1).fillna(dep_stop)
        df_cal_uid["valid_start"] = df_cal_uid["valid_start"].clip(lower=dep_start)
        df_cal_uid["valid_stop"] = df_cal_uid["valid_stop"].clip(upper=dep_stop)
        df_cal_uid = df_cal_uid[df_cal_uid["valid_start"] < df_cal_uid["valid_stop"]]

        for row in df_cal_uid.itertuples():
            df_coef = pd.read_csv(row.download_url)
            coef_dict = dict(zip(df_coef["name"], df_coef["value"]))

            # Resolve SheetRef: values (e.g. OPTAA CC_taarray / CC_tcarray)
            coef_dict = _resolve_sheetrefs(coef_dict, row.download_url)

            record = {
                "asset_uid": asset_uid,
                "calibration_date": row.cal_date,
                "valid_start": row.valid_start,
                "valid_stop": row.valid_stop,
            }
            record.update(coef_dict)
            coef_records.append(record)

    df_out = pd.DataFrame(coef_records).sort_values("calibration_date")

    if df_out.empty:
        raise ValueError("No calibration records fall within deployment windows")

    # ---------------------------
    # 4. Parse coefficient values safely
    # ---------------------------
    def parse_value(val):
        # Already a numpy array (resolved SheetRef) — pass through
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                return np.array(ast.literal_eval(val), dtype=float)
            try:
                return float(val)
            except ValueError:
                return val
        return val

    for col in df_out.columns:
        if col not in ["asset_uid", "calibration_date", "valid_start", "valid_stop"]:
            df_out[col] = df_out[col].apply(parse_value)

    # ---------------------------
    # 5. Build xarray Dataset
    # ---------------------------
    coords = {
        "calibration": df_out["calibration_date"].values,
        "valid_start": ("calibration", df_out["valid_start"].values),
        "valid_stop": ("calibration", df_out["valid_stop"].values),
        "asset_uid": ("calibration", df_out["asset_uid"].values),
    }

    def _pad_to_shape(arrays, target_shape):
        """Pad a list of ndarrays to target_shape with NaN, return stacked array."""
        out = np.full((len(arrays),) + target_shape, np.nan, dtype=float)
        for i, arr in enumerate(arrays):
            idx = tuple(slice(0, s) for s in arr.shape)
            out[(i,) + idx] = arr
        return out

    data_vars = {}
    for col in df_out.columns:
        if col in ["asset_uid", "calibration_date", "valid_start", "valid_stop"]:
            continue
        values = df_out[col].values
        first = values[0]

        if isinstance(first, np.ndarray) and first.ndim == 2:
            # 2-D array coefficient (e.g. OPTAA CC_taarray / CC_tcarray)
            shapes = [v.shape for v in values]
            target = tuple(max(s[i] for s in shapes) for i in range(2))
            if len(set(shapes)) == 1:
                stacked = np.stack(values)
            else:
                stacked = _pad_to_shape(values, target)
            data_vars[col] = (("calibration", f"{col}_row", f"{col}_col"), stacked)

        elif isinstance(first, np.ndarray) and first.ndim == 1:
            # 1-D array coefficient
            shapes = [v.shape for v in values]
            if len(set(shapes)) == 1:
                stacked = np.stack(values)
            else:
                target = (max(s[0] for s in shapes),)
                stacked = _pad_to_shape(values, target)
            data_vars[col] = (("calibration", f"{col}_index"), stacked)

        else:
            # Scalar coefficient
            data_vars[col] = (("calibration",), values.astype(float))

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds


def calibration_index(cals, timestamps):
    """
    Calibration row valid at each time (valid_start <= t < valid_stop), -1 where no
    window covers it. get_calibration_dataset windows don't overlap; if they do, the
    latest-starting window that has begun is used, or -1 if it has already ended.
    """
    start, stop = cals.valid_start.values, cals.valid_stop.values
    order = np.argsort(start)
    t = np.asarray(timestamps)
    pos = np.searchsorted(start[order], t, side="right") - 1
    row = order[np.maximum(pos, 0)]
    return np.where((pos >= 0) & (t < stop[row]), row, -1).astype("int32")


def broadcast_calibrations(cals, timestamps):
    """
    Memory-safe calibration lookup (no full broadcast).
    Returns calibration tables + index mapping. The index is -1 where no calibration
    covers a time; every table ends in a NaN row, so table[index] is NaN there rather
    than the last calibration.
    """
    tables = {
        var: np.concatenate([cals[var].values, np.full((1,) + cals[var].shape[1:], np.nan)])
        for var in cals.data_vars
    }
    return {
        "index": calibration_index(cals, timestamps),  # shape (n_time,)
        "tables": tables,                              # small calibration arrays, + NaN row
    }

# -------------------------------------------------------------------------------------
# Loading helpers for calculation dictionaries from CSVs and creating FUNCTION_REGISTRY
# -------------------------------------------------------------------------------------

# Helper to parse kwargs in CSV
def parse_kwargs(kwargs_val):
    if pd.isna(kwargs_val) or kwargs_val == "":
        return {}
    out = {}
    for pair in str(kwargs_val).split(";"):
        k, v = pair.split("=")
        out[k.strip()] = ast.literal_eval(v.strip())
    return out

# Load calculateStrings.csv → CALC_META
def load_calc_metadata(calc_csv_path):
    raw = pd.read_csv(calc_csv_path).set_index("calculation").T.to_dict()
    calc_meta = {}
    for calc_name, meta in raw.items():
        calc_meta[calc_name] = {
            "function_key": meta["function_key"].strip(),
            "inputs": [x.strip() for x in str(meta["inputs"]).split("|") if x.strip()],
            "outputs": [x.strip() for x in str(meta["returnParam"]).split("|") if x.strip()],
        }
        kwargs = parse_kwargs(meta.get("kwargs"))
        if kwargs:
            calc_meta[calc_name]["kwargs"] = kwargs
    return calc_meta

# Load site_calculations.csv → SITE_CALCULATIONS
def load_site_calculations(site_csv_path, during_harvest=None):
    df = pd.read_csv(site_csv_path)
    if during_harvest is not None:
        df = df[df['runDuringHarvest'] == during_harvest]
    return {
        row["zarrFile"]: [c.strip() for c in row["calculations"].split("|")]
        for _, row in df.iterrows()
    }

# Build function registry → FUNCTION_REGISTRY
def build_function_registry(calc_meta, module_name="advanced_qaqc.calculateFunctions"):
    registry = {}
    module = importlib.import_module(module_name)
    for calc_name, meta in calc_meta.items():
        func_key = meta["function_key"]
        registry[func_key] = getattr(module, func_key)
    return registry


def load_max_coordinate_sizes(path):
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        result.setdefault(row["instrument"], {})[row["coordinate"]] = int(row["max_size"])
    return result

