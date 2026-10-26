import numpy as np
import pytest

from rca_data_tools.qaqc.decimate import _lttb_loop


def reference_lttb(data, threshold):
    """The original list-of-buckets LTTB, in plain python, as the oracle."""
    out = np.zeros((threshold, 2))
    out[0], out[-1] = data[0], data[-1]
    bins = np.array_split(data[1:-1], threshold - 2)
    for i, bucket in enumerate(bins):
        nxt = bins[i + 1] if i < len(bins) - 1 else data[-1:]
        c = [sum(nxt[:, 0]) / len(nxt), sum(nxt[:, 1]) / len(nxt)]  # sequential, like numba's np.mean
        a = out[i]
        areas = 0.5 * np.abs((a[0] - c[0]) * (bucket[:, 1] - a[1]) - (a[0] - bucket[:, 0]) * (c[1] - a[1]))
        out[i + 1] = [bucket[len(bucket) // 2, 0], bucket[np.argmax(areas), 1]]
    return out


@pytest.mark.parametrize("n,threshold", [(10, 3), (101, 50), (1000, 900), (2000, 200), (5000, 4999)])
@pytest.mark.parametrize("kind", ["noise", "nan", "const", "flags"])
def test_lttb_matches_reference(n, threshold, kind):
    rng = np.random.default_rng(n + threshold)
    t = np.cumsum(rng.integers(1, 60, n)).astype(float) * 1e9
    y = {"noise": rng.normal(size=n), "nan": rng.normal(size=n), "const": np.ones(n),
         "flags": rng.choice([1.0, 3.0, 4.0], n)}[kind]
    if kind == "nan":
        y[rng.random(n) < 0.1] = np.nan
    data = np.stack([t, y], axis=1)
    got, want = _lttb_loop(data, threshold), reference_lttb(data, threshold)
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))
