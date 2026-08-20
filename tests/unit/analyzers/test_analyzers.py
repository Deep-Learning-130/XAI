"""Analyzers turn a dataframe into un-scored, un-annotated hypotheses.

docs/10 invariant 2: nothing here may assign severity. docs/01: analyzers emit
findings, they do not judge them.
"""
import numpy as np
import pandas as pd
import pytest

from dbias.analyzers.base import Hypothesis
from dbias.analyzers.label_disparity import LabelDisparityAnalyzer
from dbias.analyzers.missingness import MissingnessAnalyzer
from dbias.analyzers.representation import RepresentationAnalyzer
from dbias.models.enums import Category, Detectability, Severity
from dbias.stats.chi_square import GofSample


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    gender = np.array(["M"] * 160 + ["F"] * 40)
    income = rng.normal(50_000, 10_000, n)
    # Income is missing far more often for F than for M: a MAR pattern.
    missing = np.where(gender == "F", rng.random(n) < 0.5, rng.random(n) < 0.05)
    income[missing] = np.nan
    return pd.DataFrame(
        {
            "gender": gender,
            "region": rng.choice(["north", "south"], n),
            "income": income,
            "hired": rng.choice([0, 1], n),
        }
    )


# --- representation ----------------------------------------------------------

def test_representation_emits_one_hypothesis_per_attribute(frame):
    out = RepresentationAnalyzer().analyze(frame, ["gender", "region"])
    assert len(out) == 2
    assert {h.finding.sensitive_attribute for h in out} == {"gender", "region"}


def test_representation_uses_a_goodness_of_fit_sample(frame):
    (hypothesis,) = RepresentationAnalyzer().analyze(frame, ["gender"])
    assert isinstance(hypothesis.sample, GofSample)
    assert sorted(hypothesis.sample.counts) == [40, 160]


def test_representation_records_group_shares(frame):
    (hypothesis,) = RepresentationAnalyzer().analyze(frame, ["gender"])
    assert hypothesis.finding.observed_values["M"] == pytest.approx(0.8)
    assert hypothesis.finding.n_per_group == {"M": 160, "F": 40}


def test_representation_has_no_target_feature(frame):
    (hypothesis,) = RepresentationAnalyzer().analyze(frame, ["gender"])
    assert hypothesis.finding.category is Category.REPRESENTATION
    assert hypothesis.finding.target_feature is None


def test_representation_skips_a_constant_attribute():
    df = pd.DataFrame({"gender": ["M"] * 50})
    assert RepresentationAnalyzer().analyze(df, ["gender"]) == []


# --- missingness -------------------------------------------------------------

def test_missingness_is_named_mar_not_mnar(frame):
    """plan.md Sec 3.7: testing a NaN mask against an *observed* attribute
    detects MAR. True MNAR is not identifiable from observed data."""
    out = MissingnessAnalyzer().analyze(frame, ["gender"])
    ids = [h.finding.id for h in out]
    assert ids and all(i.startswith("MAR_") for i in ids)
    assert not any("MNAR" in i for i in ids)


def test_missingness_only_covers_features_that_have_missing_values(frame):
    out = MissingnessAnalyzer().analyze(frame, ["gender"])
    assert {h.finding.target_feature for h in out} == {"income"}


def test_missingness_sample_is_a_two_by_k_contingency_table(frame):
    (hypothesis,) = MissingnessAnalyzer().analyze(frame, ["gender"])
    table = np.asarray(hypothesis.sample)
    assert table.shape == (2, 2)
    assert table.sum() == len(frame)


def test_missingness_never_tests_an_attribute_against_itself():
    df = pd.DataFrame({"gender": ["M", "F", None, "M"] * 25})
    out = MissingnessAnalyzer().analyze(df, ["gender"])
    assert all(h.finding.target_feature != "gender" for h in out)


# --- label disparity ---------------------------------------------------------

def test_label_disparity_emits_one_hypothesis_per_attribute(frame):
    out = LabelDisparityAnalyzer().analyze(frame, ["gender", "region"], target_col="hired")
    assert len(out) == 2
    assert all(h.finding.target_feature == "hired" for h in out)
    assert all(h.finding.category is Category.LABEL_DISPARITY for h in out)


def test_label_disparity_requires_a_target(frame):
    with pytest.raises(ValueError):
        LabelDisparityAnalyzer().analyze(frame, ["gender"])


def test_label_disparity_reports_the_positive_rate_per_group(frame):
    (hypothesis,) = LabelDisparityAnalyzer().analyze(frame, ["gender"], target_col="hired")
    assert set(hypothesis.finding.observed_values) == {"M", "F"}
    assert all(0.0 <= v <= 1.0 for v in hypothesis.finding.observed_values.values())


# --- shared invariants -------------------------------------------------------

@pytest.mark.parametrize(
    "analyzer,kwargs",
    [
        (RepresentationAnalyzer(), {}),
        (MissingnessAnalyzer(), {}),
        (LabelDisparityAnalyzer(), {"target_col": "hired"}),
    ],
)
def test_analyzers_emit_unscored_unannotated_findings(frame, analyzer, kwargs):
    for hypothesis in analyzer.analyze(frame, ["gender"], **kwargs):
        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.finding.severity is Severity.UNDETERMINED
        assert hypothesis.finding.detectability is Detectability.UNKNOWN
        assert hypothesis.finding.p_value_corrected is None


@pytest.mark.parametrize(
    "analyzer,kwargs",
    [
        (RepresentationAnalyzer(), {}),
        (MissingnessAnalyzer(), {}),
        (LabelDisparityAnalyzer(), {"target_col": "hired"}),
    ],
)
def test_findings_carry_a_p_value_and_an_effect_size_together(frame, analyzer, kwargs):
    """Global constraint: no p-value is ever returned without an effect size."""
    for hypothesis in analyzer.analyze(frame, ["gender"], **kwargs):
        assert 0.0 <= hypothesis.finding.p_value_raw <= 1.0
        assert hypothesis.finding.effect_size_value >= 0.0
