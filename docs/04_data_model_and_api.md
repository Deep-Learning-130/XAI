# Data Model and Internal APIs

To ensure loose coupling between the statistical engines and the frontend reporting tools, all modules must communicate using standardized dataclasses and interfaces.

## 1. The `Finding` Dataclass

The `Finding` object is the universal contract representing a detected issue.

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Finding:
    id: str                         # Unique ID, e.g., "MNAR_INCOME_GENDER"
    category: str                   # Enum: "Missingness", "Representation", "Label_Disparity", etc.
    severity: str                   # Enum: "Informational", "Low", "Medium", "High", "Critical"
    
    sensitive_attribute: str        # e.g., "Gender"
    target_feature: str             # e.g., "Income" (or None for Representation)
    
    metric_name: str                # e.g., "Missingness Rate Disparity"
    observed_values: Dict[str, Any] # e.g., {"Male": 0.02, "Female": 0.15}
    
    # Statistical Evidence
    statistical_test: str           # e.g., "Chi-Square Test of Independence"
    p_value_raw: float
    p_value_corrected: float        # Post BH-FDR correction
    effect_size_metric: str         # e.g., "Cramer's V"
    effect_size_value: float
    is_significant: bool            # True if p_value_corrected < alpha
    
    # Human-Readable Content
    evidence_text: str              # Auto-generated factual statement
    potential_impact: str           # Auto-generated impact warning
    recommendations: List[str]      # Mitigation steps
```

## 2. Base Analyzer API

All Bias Analyzers must inherit from a common Abstract Base Class.

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class BaseAnalyzer(ABC):
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, sensitive_cols: List[str], target_col: str = None) -> List[Finding]:
        """
        Executes analysis on the dataframe.
        Must return a list of Finding objects (severity can be undetermined at this stage).
        """
        pass
```

## 3. Dataset Profile Schema

A lightweight schema definition to track column types.

```python
@dataclass
class DatasetProfile:
    total_rows: int
    categorical_cols: List[str]
    continuous_cols: List[str]
    boolean_cols: List[str]
    sensitive_cols: List[str]
    target_col: str
```
