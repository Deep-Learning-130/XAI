"""Dataset ingestion and loaders for benchmark datasets."""

import os
from pathlib import Path
import pandas as pd
import urllib.request

DATA_DIR = Path(__file__).parent.parent.parent.parent / "datasets"

def fetch_adult() -> pd.DataFrame:
    """Fetch the Adult Census Income dataset from UCI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "adult.csv"
    
    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        urllib.request.urlretrieve(url, csv_path)
        
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", 
        "marital-status", "occupation", "relationship", "race", "sex", 
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    
    df = pd.read_csv(csv_path, names=columns, skipinitialspace=True, na_values="?")
    return df

def fetch_compas() -> pd.DataFrame:
    """Fetch the ProPublica COMPAS dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "compas-scores-two-years.csv"
    
    if not csv_path.exists():
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        urllib.request.urlretrieve(url, csv_path)
        
    return pd.read_csv(csv_path)
