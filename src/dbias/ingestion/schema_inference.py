"""Infer dataset schema (categorical vs numerical) for the profile."""

import pandas as pd
from dbias.models.enums import VariableKind

def infer_variable_kind(series: pd.Series) -> VariableKind:
    """Infer whether a series is categorical or continuous."""
    # If it's boolean or object (string), it's categorical
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return VariableKind.CATEGORICAL
        
    # If it's numeric but has very few unique values, treat as categorical
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return VariableKind.CATEGORICAL
        return VariableKind.CONTINUOUS
        
    return VariableKind.CATEGORICAL

def infer_schema(df: pd.DataFrame) -> dict[str, VariableKind]:
    """Return a mapping of column names to inferred variable kinds."""
    return {col: infer_variable_kind(df[col]) for col in df.columns}
