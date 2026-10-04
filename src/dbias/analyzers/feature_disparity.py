"""Does a feature fall differently across sensitive groups?"""
from collections.abc import Mapping, Sequence

import pandas as pd

from dbias.analyzers.base import BaseAnalyzer, Hypothesis, excluded_features, slug
from dbias.models.enums import Category, EffectSizeMetric
from dbias.models.finding import Finding
from dbias.stats.chi_square import chi_square_test


class FeatureDisparityAnalyzer(BaseAnalyzer):
    def __init__(self, intersections: Mapping[str, Sequence[str]] | None = None) -> None:
        # Intersection column -> the attributes it was built from.
        self.intersections = intersections or {}

    def analyze(
        self,
        df: pd.DataFrame,
        sensitive_cols: list[str],
        target_col: str | None = None,
    ) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        for attribute in sensitive_cols:
            groups = df[attribute]
            if groups.nunique(dropna=True) < 2:
                continue

            for feature in df.columns:
                if feature == target_col or feature in excluded_features(
                    attribute, self.intersections
                ):
                    continue
                    
                # We only test categorical features (or discrete features with few levels)
                # to avoid massive contingency tables. Continuous features (KS/MWU) are
                # deferred to future scope (plan.md).
                if df[feature].nunique(dropna=True) > 20:
                    continue
                    
                if df[feature].nunique(dropna=True) < 2:
                    continue

                table = pd.crosstab(df[feature], groups)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                result = chi_square_test(table.to_numpy())
                
                finding = Finding(
                    id=f"DISP_{slug(feature)}_{slug(attribute)}",
                    category=Category.FEATURE_DISPARITY,
                    sensitive_attribute=attribute,
                    target_feature=feature,
                    metric_name=f"Distribution of {feature} across {attribute}",
                    observed_values={},  # Omitted for brevity on multi-level features
                    statistical_test="Chi-square test of independence",
                    p_value_raw=result.p_value,
                    effect_size_metric=EffectSizeMetric.CRAMERS_V,
                    effect_size_value=result.effect_size,
                    n_per_group={
                        str(level): int(table[level].sum()) for level in table.columns
                    },
                )
                out.append(Hypothesis(finding=finding, sample=table.to_numpy()))
        return out
