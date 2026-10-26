"""decimate.py

This module contains code for performing LTTB decimation.

"""
import gc
import math
import numba
import numpy as np
import pandas as pd
from typing import Optional
import logging
import xarray as xr
import dask.array as darray

from functools import reduce


@numba.njit
def _lttb_loop(data: np.ndarray, threshold: int) -> np.ndarray:
    """Largest Triangle Three Buckets, after https://github.com/devoxi/lttb-py.

    Keeps the first and last points; for each of the threshold-2 buckets, takes
    the time at the bucket middle and the value of the point forming the largest
    triangle with the previous output point and the next bucket's mean.

    Same output, bit for bit, as the original list-of-buckets version: buckets sized like np.array_split, the next-bucket
    mean summed sequentially as numba's np.mean does, the area expression in
    the same operation order, and argmax keeping the first maximum (or the
    first NaN, as np.argmax does). It just walks indices instead of allocating
    a list of bucket views plus temporaries for every bucket, which dominated
    the runtime when buckets hold only a point or two.
    """
    n_bins = threshold - 2
    n = data.shape[0]
    out = np.zeros((threshold, 2))
    out[0, 0] = data[0, 0]
    out[0, 1] = data[0, 1]
    out[threshold - 1, 0] = data[n - 1, 0]
    out[threshold - 1, 1] = data[n - 1, 1]

    # np.array_split of the n-2 interior points: the first `extra` bins get one more
    n_items = n - 2
    base = n_items // n_bins
    extra = n_items % n_bins

    start = 1
    for i in range(n_bins):
        size = base + 1 if i < extra else base
        end = start + size

        # mean of the next bin (the last bin looks ahead to the final point)
        if i < n_bins - 1:
            next_end = end + (base + 1 if i + 1 < extra else base)
            c0 = 0.0
            c1 = 0.0
            for j in range(end, next_end):
                c0 += data[j, 0]
                c1 += data[j, 1]
            c0 /= next_end - end
            c1 /= next_end - end
        else:
            c0 = data[n - 1, 0]
            c1 = data[n - 1, 1]

        a0 = out[i, 0]
        a1 = out[i, 1]
        best = start
        best_area = -1.0
        for j in range(start, end):
            area = 0.5 * np.abs((a0 - c0) * (data[j, 1] - a1) - (a0 - data[j, 0]) * (c1 - a1))
            if np.isnan(area):  # np.argmax returns the first NaN
                best = j
                break
            if area > best_area:
                best_area = area
                best = j

        out[i + 1, 0] = data[start + size // 2, 0]  # middle of the bucket
        out[i + 1, 1] = data[best, 1]
        start = end
    return out


class LttbException(Exception):
    pass


def _perform_decimation(ds, threshold):
    time_da = ds.time.astype(int)
    cols = [time_da.name, ds.name]
    data = darray.stack([time_da.data, ds.data], axis=1).compute()
    try:
        decdata = _lttb_loop(data, threshold)
    except Exception as e:
        raise LttbException(e)
    del ds
    gc.collect()
    return pd.DataFrame(decdata, columns=cols)


def downsample(
    raw_ds: xr.Dataset,
    threshold: int,
    logger: Optional[logging.Logger] = None,
    max_points_factor: int = 10,
) -> pd.DataFrame:
    """
    Performs downsampling on the dataset.

    Parameters
    ----------
    raw_ds : xr.Dataset
        The dataset to be decimated.
    threshold : int
        The threshold for decimation,
        total number of data points at the end.
    logger : logging.Logger
        Logger instance to be used for logging
    max_points_factor : int
        Cap, as a multiple of ``threshold``, on how many points are
        materialized per variable before LTTB. Oversized (e.g. 365-day
        velocity) arrays are lazily strided down to this cap first so the
        full array never lands in memory. LTTB then runs on the reduced
        array to preserve visually significant points.

    Returns
    -------
    pd.DataFrame
        The decimated data as pandas dataframe.
    """
    if logger is None:
        from loguru import logger
    logger.info("Get list of data arrays")
    da_list = (raw_ds[var] for var in raw_ds)

    cap = threshold * max_points_factor
    df_list = []
    for da in da_list:
        # Lazily stride down before materializing so huge dask-backed arrays
        # never land in memory all at once (prevents OOM on long spans).
        n_time = da.time.size
        if n_time > cap:
            stride = math.ceil(n_time / cap)
            logger.info(
                f"Pre-striding {da.name}: {n_time} -> ~{n_time // stride} points "
                f"(stride {stride}) before LTTB"
            )
            da = da.isel(time=slice(None, None, stride))

        logger.info(f"Executing decimation for {da.name}")
        decdf = _perform_decimation(da, threshold)
        df_list.append(decdf)

        del decdf
        gc.collect()
    logger.info("Decimation process completed.")

    logger.info("Creating decimated dataframe.")
    # Every variable shares the time axis and stride, so LTTB gives them all
    # the same bucket-middle times: join by column instead of a chained merge,
    # which recopied the growing frame for every variable. Same frame as the
    # merge; fall back to it if the times ever differ.
    times = [df["time"].to_numpy() for df in df_list]
    if all(np.array_equal(times[0], t) for t in times[1:]) and pd.Index(times[0]).is_unique:
        final_df = pd.concat([df_list[0]] + [df.drop(columns="time") for df in df_list[1:]], axis=1)
    else:
        final_df = reduce(
            lambda left, right: pd.merge(left, right, on="time"), df_list
        )
    final_df['time'] = pd.to_datetime(final_df['time'])

    return final_df
