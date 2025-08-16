
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=16)
def _csv_cache(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_csv_cached(path: str) -> pd.DataFrame:
    # wrapper to avoid passing Path objects into cache
    return _csv_cache(str(path))
