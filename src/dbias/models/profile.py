"""Lightweight column-kind bookkeeping for a loaded dataset."""
from dataclasses import dataclass, field


@dataclass
class DatasetProfile:
    total_rows: int
    categorical_cols: list[str] = field(default_factory=list)
    continuous_cols: list[str] = field(default_factory=list)
    boolean_cols: list[str] = field(default_factory=list)
    sensitive_cols: list[str] = field(default_factory=list)
    target_col: str | None = None
