__author__ = "Christian Marius Lillelund"
__author_email__ = "chr1000@gmail.com"

from importlib import resources
from typing import Dict, Callable, Tuple

import numpy as np
import pandas as pd
import shap
from sksurv import datasets

resource_package = __name__


def convert_to_structured(T, E) -> np.typing.NDArray:
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)


def load_seer_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    resource = resources.open_binary(resource_package, "seer.csv")
    data = pd.read_csv(resource)

    outcomes = data.copy()
    outcomes["event"] = data["Status"]
    outcomes["time"] = data["Survival Months"]
    outcomes = outcomes[["event", "time"]]
    outcomes.loc[outcomes["event"] == "Alive", ["event"]] = 0
    outcomes.loc[outcomes["event"] == "Dead", ["event"]] = 1

    data = data.drop(["Status", "Survival Months"], axis=1)

    X = pd.DataFrame(data)
    y = convert_to_structured(outcomes["time"], outcomes["event"])
    return X, y


def load_nhanes_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    nhanes_X, nhanes_y = shap.datasets.nhanesi()
    X = pd.DataFrame(nhanes_X)
    event = np.array([True if x > 0 else False for x in nhanes_y])
    time = np.array(abs(nhanes_y))
    y = convert_to_structured(time, event)
    return X, y


def load_support_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    resource = resources.open_binary(resource_package, "support.feather")
    data = pd.read_feather(resource)

    outcomes = data.copy()
    outcomes["event"] = data["event"]
    outcomes["time"] = data["duration"]
    outcomes = outcomes[["event", "time"]]

    feats = [
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
    ]

    X = pd.DataFrame(data[feats])
    y = convert_to_structured(outcomes["time"], outcomes["event"])
    return X, y


def load_aids_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    X, y = datasets.load_aids()
    X = pd.DataFrame(X)
    y = convert_to_structured(y["time"], y["censor"])
    return X, y


def load_veterans_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    X, y = datasets.load_veterans_lung_cancer()
    X = pd.DataFrame(X)
    y = convert_to_structured(y["Survival_in_days"], y["Status"])
    return X, y


def load_whas500_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    X, y = datasets.load_whas500()
    X = pd.DataFrame(X)
    y = convert_to_structured(y["lenfol"], y["fstat"])
    return X, y


def load_flchain_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    X, y = datasets.load_flchain()
    X = pd.DataFrame(X)
    y = convert_to_structured(y["futime"], y["death"])
    return X, y


def load_gbsg2_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    X, y = datasets.load_gbsg2()
    X = pd.DataFrame(X)
    y = convert_to_structured(y["time"], y["cens"])
    return X, y


def load_metabric_dataset() -> Tuple[pd.DataFrame, np.typing.NDArray]:
    resource = resources.open_binary(resource_package, "metabric.feather")
    data = pd.read_feather(resource)

    outcomes = data.copy()
    outcomes["event"] = data["event"]
    outcomes["time"] = data["duration"]
    outcomes = outcomes[["event", "time"]]

    num_feats = ["x0", "x1", "x2", "x3", "x8"] + ["x4", "x5", "x6", "x7"]

    X = pd.DataFrame(data[num_feats])
    y = convert_to_structured(outcomes["time"], outcomes["event"])
    return X, y


_DATASETS: Dict[str, Callable[[], Tuple[pd.DataFrame, np.typing.NDArray]]] = {
    "seer": load_seer_dataset,
    "nhanes": load_nhanes_dataset,
    "support": load_support_dataset,
    "aids": load_aids_dataset,
    "veterans": load_veterans_dataset,
    "whas500": load_whas500_dataset,
    "flchain": load_flchain_dataset,
    "gbsg2": load_gbsg2_dataset,
    "metabric": load_metabric_dataset,
}


def _load_dataset(name: str) -> Tuple[pd.DataFrame, np.typing.NDArray]:
    loader = _DATASETS.get(name)
    if loader is not None:
        return loader()

    raise ValueError(
        f"Unknown data set name {name}, choose one of {', '.join(_DATASETS)}"
    )
