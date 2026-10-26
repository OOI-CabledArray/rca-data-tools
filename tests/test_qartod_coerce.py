import numpy as np
import pytest
import xarray as xr

from rca_data_tools.qaqc import utils


class ListLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        pass


@pytest.fixture
def logger(monkeypatch):
    log = ListLogger()
    monkeypatch.setattr(utils, "select_logger", lambda: log)
    return log


def make_ds(values, tests, chunks=None):
    var = xr.DataArray(np.array(values), dims="time", attrs={"tests_executed": tests})
    ds = xr.Dataset({"temp_qartod_executed": var}, coords={"time": np.arange(len(values))})
    return ds.chunk({"time": chunks}) if chunks else ds


@pytest.mark.parametrize("chunks", [None, 2])
def test_valid_flags_decode_unchanged(logger, chunks):
    ds = utils.coerce_qartod_executed_to_int(make_ds(["13", "41", "19"], "gross_range_test, climatology_test", chunks))
    assert ds["temp_qartod_executed_gross_range_test"].values.tolist() == [1, 4, 1]
    assert ds["temp_qartod_executed_climatology_test"].values.tolist() == [3, 1, 9]
    assert "temp_qartod_executed" not in ds
    assert logger.warnings == []


@pytest.mark.parametrize("chunks", [None, 2])
def test_invalid_flags_become_not_evaluated_with_warning(logger, chunks):
    # '' is a missing flag; 'n' is a 'nan' fill squeezed into a one-character string
    ds = utils.coerce_qartod_executed_to_int(make_ds(["1", "", "n", "4", "n"], "gross_range_test", chunks))
    assert ds["temp_qartod_executed_gross_range_test"].values.tolist() == [1, 2, 2, 4, 2]
    assert logger.warnings, "expected a warning for the invalid values"
    text = " ".join(logger.warnings)
    assert "Invalid QARTOD flag value" in text
    assert "temp_qartod_executed" in text and "gross_range_test" in text
    assert "not evaluated" in text
    assert sum(int(w.split(": ")[1].split(" ")[0]) for w in logger.warnings) == 3  # every bad point counted


def test_invalid_character_in_one_test_only(logger):
    ds = utils.coerce_qartod_executed_to_int(make_ds(["1n", "14"], "gross_range_test, climatology_test"))
    assert ds["temp_qartod_executed_gross_range_test"].values.tolist() == [1, 1]
    assert ds["temp_qartod_executed_climatology_test"].values.tolist() == [2, 4]
    assert len(logger.warnings) == 1 and "climatology_test" in logger.warnings[0]
