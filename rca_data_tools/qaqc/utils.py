import ast
import numpy as np
import os
import pandas as pd
import re
import requests
import xarray as xr
import importlib

def select_logger():
    from prefect import get_run_logger
    try:
        logger = get_run_logger()
    except:
        from loguru import logger
    
    return logger


def coerce_qartod_executed_to_int(ds):
    logger = select_logger()

    qartod_executed_vars = [var for var in ds.variables if 'qartod_executed' in var]
    for var in qartod_executed_vars:
        executed_tests = ds[var].tests_executed.replace(' ', '').split(',')
    
        for i, test in enumerate(executed_tests):
            test_var_name = f"{var}_{test}"
            ds[test_var_name] = ds[var].str[i].astype(int)

        ds = ds.drop(var)
    logger.info(f"ds size post coercion: {ds.nbytes}")
    return ds


def get_s3_kwargs():
    aws_key = os.environ.get("AWS_KEY")
    aws_secret = os.environ.get("AWS_SECRET")
    
    s3_kwargs = {'key': aws_key, 'secret': aws_secret}
    return s3_kwargs


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


def broadcast_calibrations(cals, timestamps):
    """
    Broadcast calibration coefficients to a large time series efficiently.

    Parameters
    ----------
    cals : xr.Dataset
        Calibration dataset from get_calibration_dataset()
    timestamps : pd.DatetimeIndex or np.ndarray of datetime64
        Measurement timestamps

    Returns
    -------
    dict
        Keys = coefficient names
        Values = numpy arrays aligned to timestamps:
        - scalar coefficients    → shape (n,)
        - 1-D array coefficients → shape (n, array_length)
        - 2-D array coefficients → shape (n, n_rows, n_cols)
    """
    n = len(timestamps)
    out = {}

    intervals = pd.IntervalIndex.from_arrays(
        cals.valid_start.values, cals.valid_stop.values, closed="left"
    )
    indices = intervals.get_indexer(timestamps)

    for var in cals.data_vars:
        values = cals[var].values   # shape: (n_cal,), (n_cal, k), or (n_cal, r, c)
        mask = indices >= 0

        if values.ndim == 1:
            arr = np.full(n, np.nan, dtype=float)
            arr[mask] = values[indices[mask]]

        elif values.ndim == 2:
            arr = np.full((n, values.shape[1]), np.nan, dtype=float)
            arr[mask, :] = values[indices[mask], :]

        elif values.ndim == 3:
            arr = np.full((n, values.shape[1], values.shape[2]), np.nan, dtype=float)
            arr[mask, :, :] = values[indices[mask], :, :]

        else:
            raise ValueError(f"Unsupported ndim {values.ndim} for variable '{var}'")

        out[var] = arr

    return out

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
def load_site_calculations(site_csv_path):
    df = pd.read_csv(site_csv_path)
    return {
        row["refDes"]: [c.strip() for c in row["calculations"].split("|")]
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

