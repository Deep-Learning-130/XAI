"""Does the outcome label fall differently across groups?"""
import pandas as pd

from dbias.analyzers.base import BaseAnalyzer, Hypothesis, slug
from dbias.models.enums import Category, EffectSizeMetric
from dbias.models.finding import Finding
from dbias.stats.chi_square import chi_square_test


class LabelDisparityAnalyzer(BaseAnalyzer):
    def analyze(
        self,
        df: pd.DataFrame,
        sensitive_cols: list[str],
        target_col: str | None = None,
    ) -> list[Hypothesis]:
        if target_col is None:
            raise ValueError("label disparity needs a target column to test against")
        if target_col not in df.columns:
            raise ValueError(f"target column {target_col!r} is not in the dataframe")

        out: list[Hypothesis] = []
        for attribute in sensitive_cols:
            if attribute == target_col:
                continue
            groups = df[attribute]
            if groups.nunique(dropna=True) < 2:
                continue
            if df[target_col].nunique(dropna=True) < 2:
                continue

            table = pd.crosstab(df[target_col], groups)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            result = chi_square_test(table.to_numpy())
            # Share of each group falling in the highest label level -- the
            # "positive outcome" rate under the usual 0/1 encoding.
            shares = table / table.sum(axis=0)
            top_level = table.index[-1]

            finding = Finding(
                id=f"LABEL_{slug(target_col)}_{slug(attribute)}",
                category=Category.LABEL_DISPARITY,
                sensitive_attribute=attribute,
                target_feature=target_col,
                metric_name=f"{target_col}={top_level} rate across groups",
                observed_values={
                    str(level): float(shares.loc[top_level, level])
                    for level in table.columns
                },
                statistical_test="Chi-square test of independence",
                p_value_raw=result.p_value,
                effect_size_metric=EffectSizeMetric.CRAMERS_V_CORRECTED,
                effect_size_value=result.effect_size,
                n_per_group={
                    str(level): int(table[level].sum()) for level in table.columns
                },
            )
            out.append(Hypothesis(finding=finding, sample=table.to_numpy()))
        return out
