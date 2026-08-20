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
