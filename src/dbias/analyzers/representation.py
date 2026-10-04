"""Are the groups present in the proportions they should be?"""
import pandas as pd

from dbias.analyzers.base import BaseAnalyzer, Hypothesis, slug
from dbias.models.enums import Category, EffectSizeMetric
from dbias.models.finding import Finding
from dbias.stats.chi_square import GofSample, chi_square_goodness_of_fit


class RepresentationAnalyzer(BaseAnalyzer):
    """Goodness of fit of group shares against a reference distribution.

    The reference defaults to uniform, which is a convention and not a claim
    about any population. Where true population shares are known, pass them in
    `reference` -- the audit is only as meaningful as the reference it uses.
    """

    def __init__(self, reference: dict[str, dict[str, float]] | None = None) -> None:
        self.reference = reference or {}

    def analyze(
        self,
        df: pd.DataFrame,
        sensitive_cols: list[str],
        target_col: str | None = None,
    ) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        for attribute in sensitive_cols:
            counts = df[attribute].value_counts(dropna=True)
            reference = self.reference.get(attribute)
            expected_probs = None
            if reference:
                observed = {str(level) for level in counts.index}
                expected = [level for level, share in reference.items() if share > 0]
                unknown = sorted(observed - set(expected))
                if unknown:
                    raise ValueError(
                        f"reference for {attribute!r} gives no positive share to "
                        f"observed level(s) {unknown}"
                    )
                # A group the reference expects but the data lacks is the most
                # extreme under-representation there is; it is counted as zero,
                # never dropped. A level the reference gives no share is not
                # expected at all, so it is not a group here.
                counts.index = counts.index.map(str)
                counts = counts.reindex(expected, fill_value=0)
                expected_probs = tuple(float(reference[level]) for level in counts.index)

            if len(counts) < 2 or counts.sum() == 0:
                # A constant or empty attribute has no shares to compare.
                continue
            result = chi_square_goodness_of_fit(counts.to_numpy(), expected_probs)
            total = int(counts.sum())

            finding = Finding(
                id=f"REPRESENTATION_{slug(attribute)}",
                category=Category.REPRESENTATION,
                sensitive_attribute=attribute,
                target_feature=None,
                metric_name="Group share vs reference distribution",
                observed_values={
                    str(level): count / total for level, count in counts.items()
                },
                statistical_test="Chi-square goodness of fit",
                p_value_raw=result.p_value,
                effect_size_metric=EffectSizeMetric.COHENS_W,
                effect_size_value=result.effect_size,
                n_per_group={
                    str(level): int(count) for level, count in counts.items()
                },
            )
            sample = GofSample(
                counts=tuple(float(c) for c in counts.to_numpy()),
                expected_probs=expected_probs,
            )
            out.append(Hypothesis(finding=finding, sample=sample))
        return out
