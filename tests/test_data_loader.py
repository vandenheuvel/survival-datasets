import pytest
import sklearn.utils as skut

from survdata import AVAILABLE_DATASETS, load_dataset

@pytest.mark.parametrize("name", AVAILABLE_DATASETS)
def test_load_dataset(name):
    nans_allowed = {
        "flchain",
        "nhanes",
    }

    check_args = {}
    if name in nans_allowed:
        check_args["ensure_all_finite"] = "allow-nan"

    X, y = load_dataset(name)
    skut.check_X_y(X, y, dtype=None, **check_args)
