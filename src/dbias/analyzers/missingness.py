"""Does a feature go missing more often for some groups than others?

Naming. This detects **MAR** -- missingness that depends on an *observed*
attribute. It is not MNAR. True MNAR means missingness depends on the
unobserved value itself, which is not identifiable from observed data at all;
docs/03 Sec 2 mislabels the test and plan.md Sec 3.7 corrects it. Finding ids
say MAR so the error does not propagate into the report.
"""
from collections.abc import Mapping, Sequence

import pandas as pd

from dbias.analyzers.base import BaseAnalyzer, Hypothesis, excluded_features, slug
from dbias.models.enums import Category, EffectSizeMetric
from dbias.models.finding import Finding
from dbias.stats.chi_square import chi_square_test


class MissingnessAnalyzer(BaseAnalyzer):
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
                if feature in excluded_features(attribute, self.intersections):
                    continue
                mask = df[feature].isna()
                if not mask.any() or mask.all():
                    # No variation in the mask: there is no hypothesis here.
                    continue

                table = pd.crosstab(mask, groups)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                result = chi_square_test(table.to_numpy())
                rates = mask.groupby(groups, observed=True).mean()

                finding = Finding(
                    id=f"MAR_{slug(feature)}_{slug(attribute)}",
                    category=Category.MISSINGNESS,
                    sensitive_attribute=attribute,
                    target_feature=feature,
                    metric_name="Missingness rate disparity",
                    observed_values={
                        str(level): float(rate) for level, rate in rates.items()
                    },
                    statistical_test="Chi-square test of independence",
                    p_value_raw=result.p_value,
                    effect_size_metric=EffectSizeMetric.CRAMERS_V,
                    effect_size_value=result.effect_size,
                    n_per_group={
                        str(level): int(count)
                        for level, count in groups.value_counts().items()
                    },
                )
                out.append(Hypothesis(finding=finding, sample=table.to_numpy()))
        return out
