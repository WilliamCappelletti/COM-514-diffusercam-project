"""I/O fuctions for data and results"""
from pathlib import Path
from typing import Optional

import pandas as pd

RESULTS_COLS = ["img_name", "objective", "reg", "reg_lambda", "reg_val", "nb_trials"]


def init_results(res_path: Optional[Path] = None) -> pd.DataFrame:
    """Load results if existing or init empty df with specified columns"""
    if res_path is None or not res_path.is_file():
        return pd.DataFrame(columns=RESULTS_COLS)
    elif res_path.suffix == ".csv":
        df = pd.read_csv(res_path, index_col=0)

        if not (set(df.columns) == set(RESULTS_COLS) and len(df.columns) == len(RESULTS_COLS)):
            raise IOError(f"Results df has columns {df.columns.tolist()}, expected {RESULTS_COLS}")

        return df
    else:
        raise ValueError(f"res_path must be csv if provided, got {res_path}")
