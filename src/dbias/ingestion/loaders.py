"""Dataset ingestion and loaders for benchmark datasets."""

import os
from pathlib import Path
import pandas as pd
import urllib.request


def data_dir() -> Path:
    """Where downloaded benchmarks are cached.

    `DBIAS_DATA_DIR` overrides; otherwise `./datasets` under the working
    directory. The location is never derived from where the package itself is
    installed, which inside site-packages would write outside the project.
    """
    return Path(os.environ.get("DBIAS_DATA_DIR", Path.cwd() / "datasets"))


def _download(url: str, path: Path) -> None:
    """Fetch `url` to `path` atomically.

    A download interrupted midway must not leave a truncated file at `path`,
    or every later call would skip the download and parse the fragment.
    """
    partial = path.with_name(path.name + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def fetch_adult() -> pd.DataFrame:
    """Fetch the Adult Census Income dataset from UCI."""
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "adult.csv"

    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        _download(url, csv_path)

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df = pd.read_csv(csv_path, names=columns, skipinitialspace=True, na_values="?")
    return df

def fetch_compas() -> pd.DataFrame:
    """Fetch the ProPublica COMPAS dataset."""
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "compas-scores-two-years.csv"

    if not csv_path.exists():
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        _download(url, csv_path)

    return pd.read_csv(csv_path)
