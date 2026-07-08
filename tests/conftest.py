import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, None, 22, 38],
        "salary": [50000, 60000, 70000, 80000, 90000, 55000, None, 75000],
        "department": ["eng", "eng", "sales", "sales", "eng", "hr", "sales", None],
        "experience": [2, 5, 8, 12, 15, 3, 1, 10],
        "target": [0, 1, 1, 0, 1, 0, 0, 1],
    })


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path, index=False)
    return str(path)
