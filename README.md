# survival-datasets

## Installation

Simply install via pip:
```
pip install survival-datasets
```

## Datasets

To list the available data sets, see the of dataset names:
```python
from survdata import AVAILABLE_DATASETS
print(AVAILABLE_DATASETS)
```

## Examples

Import the datasets module from the package and load your dataset of choice:
```python
from survdata import datasets

X, y = datasets.load_seer_dataset()
```
