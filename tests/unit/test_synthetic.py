"""Synthetic datasets with known ground truth.

These generators are the only place in the project where the true effect is
known, which makes them the only place the tool's verdicts can be checked
against reality rather than against itself (plan.md Sec 3.3).
"""
import numpy as np
import pytest

from dbias.synthetic import make_audit_scenario, make_missingness_frame


def test_frame_has_the_requested_number_of_rows():
    df = make_missingness_frame(n=500, group_shares={"a": 0.5, "b": 0.5}, seed=1)
    assert len(df) == 500


def test_group_shares_are_honoured_exactly():
    df = make_missingness_frame(
        n=1000, group_shares={"a": 0.8, "b": 0.2}, seed=1
    )
    counts = df["group"].value_counts()
    assert counts["a"] == 800
    assert counts["b"] == 200


def test_generation_is_deterministic_under_a_fixed_seed():
    a = make_missingness_frame(n=200, group_shares={"a": 0.5, "b": 0.5}, seed=42)
    b = make_missingness_frame(n=200, group_shares={"a": 0.5, "b": 0.5}, seed=42)
    assert a.equals(b)


def test_different_seeds_produce_different_frames():
    a = make_missingness_frame(n=200, group_shares={"a": 0.5, "b": 0.5}, seed=1)
    b = make_missingness_frame(n=200, group_shares={"a": 0.5, "b": 0.5}, seed=2)
    assert not a.equals(b)


def test_injected_missingness_rates_are_reproduced_in_the_data():
    df = make_missingness_frame(
        n=4000,
        group_shares={"a": 0.5, "b": 0.5},
        missing_rates={"a": 0.1, "b": 0.4},
        seed=7,
    )
    rates = df["value"].isna().groupby(df["group"]).mean()
    assert rates["a"] == pytest.approx(0.1, abs=0.03)
    assert rates["b"] == pytest.approx(0.4, abs=0.03)


def test_equal_missing_rates_mean_a_true_null():
    df = make_missingness_frame(
        n=4000,
        group_shares={"a": 0.5, "b": 0.5},
        missing_rates={"a": 0.2, "b": 0.2},
        seed=7,
    )
    rates = df["value"].isna().groupby(df["group"]).mean()
    assert rates["a"] == pytest.approx(rates["b"], abs=0.03)


# --- the flagship scenario ---------------------------------------------------

def test_audit_scenario_carries_its_ground_truth():
    df, truth = make_audit_scenario(seed=11)
    assert set(truth) >= {"n", "sensitive_cols", "target_col", "true_effects"}
    assert truth["n"] == len(df)


def test_audit_scenario_declares_at_least_one_genuine_disparity():
    _, truth = make_audit_scenario(seed=11)
    assert any(v > 0 for v in truth["true_effects"].values())


def test_audit_scenario_declares_at_least_one_genuine_null():
    _, truth = make_audit_scenario(seed=11)
    assert any(v == 0 for v in truth["true_effects"].values())


def test_audit_scenario_columns_are_all_present():
    df, truth = make_audit_scenario(seed=11)
    for column in truth["sensitive_cols"] + [truth["target_col"]]:
        assert column in df.columns


def test_audit_scenario_is_deterministic():
    a, _ = make_audit_scenario(seed=3)
    b, _ = make_audit_scenario(seed=3)
    assert a.equals(b)


def test_audit_scenario_has_missing_values_to_analyse():
    df, _ = make_audit_scenario(seed=11)
    assert df.isna().any().any()


def test_rejects_group_shares_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        make_missingness_frame(n=100, group_shares={"a": 0.5, "b": 0.6}, seed=1)
