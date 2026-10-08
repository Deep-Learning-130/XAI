"""The two artifacts an audit leaves behind: a JSON record and a figure."""
import json

import pytest

from dbias.audit import audit
from dbias.report.coverage_plot import plot_coverage_map
from dbias.report.json_export import to_dict, write_json
from dbias.synthetic import make_audit_scenario


@pytest.fixture(scope="module")
def result():
    df, truth = make_audit_scenario(n=600, seed=1)
    return audit(
        df,
        sensitive_cols=truth["sensitive_cols"],
        target_col=truth["target_col"],
        sesoi=0.1,
        seed=7,
        n_resamples=250,
    )


# --- the JSON record ---------------------------------------------------------

def test_report_states_the_sesoi_it_used(result):
    """plan.md Sec 3.6: verbatim, in the report, every time. A verdict of
    'nothing that matters was missed' is unreadable without it."""
    assert to_dict(result)["configuration"]["sesoi"] == pytest.approx(0.1)


def test_report_states_the_scale_the_sesoi_is_on(result):
    """0.1 is a very different claim on Cohen's w than on a rate difference."""
    assert "Cohen" in to_dict(result)["configuration"]["sesoi_scale"]


def test_report_states_alpha_and_target_power(result):
    config = to_dict(result)["configuration"]
    assert config["alpha"] == pytest.approx(0.05)
    assert config["target_power"] == pytest.approx(0.80)


def test_report_carries_risk_and_coverage_vectors(result):
    report = to_dict(result)
    assert "risk" in report["summary"]
    assert "coverage" in report["summary"]


def test_report_contains_no_aggregate_bias_score(result):
    flat = json.dumps(to_dict(result)).lower()
    assert "bias_score" not in flat
    assert "overall_score" not in flat


def test_every_finding_reports_its_detectability(result):
    for finding in to_dict(result)["findings"]:
        assert finding["detectability"] != "unknown"
        assert "minimum_detectable_effect" in finding
        assert "equivalence_verdict" in finding


def test_report_states_its_limitations(result):
    """The tool detects statistical disparities, not discrimination. Saying so
    in the artifact matters more than saying it in a paper nobody reads."""
    limitations = " ".join(to_dict(result)["limitations"]).lower()
    assert "causal" in limitations or "causation" in limitations


def test_report_is_json_serialisable(result):
    assert json.loads(json.dumps(to_dict(result)))


def test_write_json_produces_a_readable_file(result, tmp_path):
    path = write_json(result, tmp_path / "audit.json")
    assert path.exists()
    assert json.loads(path.read_text())["configuration"]["sesoi"] == pytest.approx(0.1)


def test_blind_spots_are_listed_separately(result):
    """They are the finding type most likely to be skimmed past, so they get
    their own section rather than being mixed into the feed."""
    assert "blind_spots" in to_dict(result)


# --- the figure --------------------------------------------------------------

def test_plot_writes_a_png(result, tmp_path):
    path = plot_coverage_map(result, tmp_path / "coverage.png")
    assert path.exists()
    assert path.stat().st_size > 5_000


def test_plot_rejects_an_audit_with_nothing_to_draw(tmp_path):
    import pandas as pd

    from dbias.audit import audit as run

    empty = run(
        pd.DataFrame({"g": ["a"] * 10}), sensitive_cols=["g"], sesoi=0.1, n_resamples=50
    )
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_coverage_map(empty, tmp_path / "coverage.png")


def test_blind_spot_reading_names_the_right_cause():
    """Underpowered: the MDE is the reason, in the SESOI's units. Powered but
    inconclusive: saying 'could only have caught X' would claim the test could
    see the SESOI while calling it a blind spot. The cause is read from
    detectability, never re-derived from the MDE."""
    from dbias.models.enums import Category, Detectability, EffectSizeMetric
    from dbias.models.finding import Finding
    from dbias.report.json_export import blind_spot_reading

    def f(mde_v, df_min, detectability, ci=(0.02, 0.14), sesoi=0.1):
        return Finding(
            id="T", category=Category.MISSINGNESS, sensitive_attribute="a",
            target_feature="x", metric_name="", observed_values={},
            statistical_test="", p_value_raw=0.5,
            effect_size_metric=EffectSizeMetric.CRAMERS_V, effect_size_value=0.0,
            n_per_group={}, sesoi=sesoi, df_min=df_min, minimum_detectable_effect=mde_v,
            detectability=detectability, effect_size_ci=ci,
        )

    under = blind_spot_reading(f(0.06, 4, Detectability.UNDERPOWERED))
    assert "not powered" in under and "w = 0.120" in under
    # MDE a hair under the SESOI but power below target: still underpowered.
    assert "not powered" in blind_spot_reading(f(0.099, 1, Detectability.UNDERPOWERED))
    assert "although the test was powered" in blind_spot_reading(f(0.085, 1, Detectability.ADEQUATE))
    assert "no interval was computed" in blind_spot_reading(
        f(0.2, 1, Detectability.ADEQUATE, ci=None)
    )
    assert "the SESOI" in blind_spot_reading(f(0.2, 1, Detectability.ADEQUATE, sesoi=None))


def test_a_gated_blind_spot_says_it_was_never_tested():
    from dbias.models.enums import Category, Detectability, EffectSizeMetric
    from dbias.models.finding import Finding
    from dbias.report.json_export import blind_spot_reading

    gated = Finding(
        id="T", category=Category.MISSINGNESS, sensitive_attribute="a_AND_b",
        target_feature="x", metric_name="", observed_values={}, statistical_test="",
        p_value_raw=1e-6, effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.3, n_per_group={}, sesoi=0.1, df_min=1,
        minimum_detectable_effect=0.05, detectability=Detectability.ADEQUATE,
        effect_size_ci=(0.2, 0.4), gated_by_parent=True,
    )
    assert "no parent attribute was found to have a disparity" in blind_spot_reading(gated)
    assert blind_spot_reading(gated).startswith("Not tested for significance")


def _gated(**overrides):
    from dbias.models.enums import Category, Detectability, EffectSizeMetric
    from dbias.models.finding import Finding

    kwargs = dict(
        id="T", category=Category.MISSINGNESS, sensitive_attribute="a_AND_b",
        target_feature="x", metric_name="", observed_values={}, statistical_test="",
        p_value_raw=1e-6, effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.3, n_per_group={}, sesoi=0.1, df_min=1,
        minimum_detectable_effect=0.15, detectability=Detectability.UNDERPOWERED,
        effect_size_ci=None, gated_by_parent=True,
    )
    kwargs.update(overrides)
    return Finding(**kwargs)


def test_a_gated_blind_spot_without_an_interval_does_not_claim_one():
    """Regression: the gated reason always said 'its interval did not rule
    out', also for skipped-bootstrap intersections that have no interval."""
    from dbias.report.json_export import blind_spot_reading

    reading = blind_spot_reading(_gated())
    assert "interval did not rule out" not in reading
    assert "no interval was computed" in reading
    assert "w = 0.150" in reading  # the underpowered explanation is kept


def test_a_gated_reason_does_not_claim_an_untested_parent_was_tested():
    """A parent with no finding (e.g. constant) also gates its children."""
    from dbias.report.json_export import blind_spot_reading

    assert "neither parent attribute showed" not in blind_spot_reading(_gated())
    assert "no parent attribute was found to have a disparity" in blind_spot_reading(_gated())


def test_hierarchical_configuration_says_it_failed_its_gate():
    import pandas as pd
    from dbias.audit import audit as run
    from dbias.report.json_export import to_dict

    df = pd.DataFrame({"a": ["x", "y"] * 50, "b": ["p"] * 50 + ["q"] * 50, "y": [0, 1] * 50})
    with pytest.warns(UserWarning):
        result = run(df, ["a", "b"], "y", sesoi=0.1, n_resamples=20, fdr="hierarchical")
    assert "failed its FDR calibration gate" in to_dict(result)["configuration"]["correction"]
