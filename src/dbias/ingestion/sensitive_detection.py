"""Detect potentially sensitive attributes based on column names and values.

This module only *suggests* sensitive attributes. It never makes the final decision
(plan.md and docs/10 invariant). The user must declare them.
"""
from typing import Set
import pandas as pd

SENSITIVE_KEYWORDS = {
    "race", "gender", "sex", "age", "ethnicity", "nationality", 
    "religion", "disability", "marital", "sexual_orientation",
    "native-country"
}

def suggest_sensitive_attributes(df: pd.DataFrame) -> Set[str]:
    """Suggest columns that might be protected/sensitive attributes."""
    suggestions = set()
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in SENSITIVE_KEYWORDS):
            suggestions.add(col)
            
    return suggestions
