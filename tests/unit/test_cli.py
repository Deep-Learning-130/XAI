"""The command line surface."""
import json

import pytest

from dbias.cli import main
from dbias.synthetic import make_audit_scenario


@pytest.fixture
def csv_path(tmp_path):
    df, _ = make_audit_scenario(n=400, seed=1)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_audit_writes_json_and_a_figure(csv_path, tmp_path):
    out = tmp_path / "out"
    code = main([
        "audit", str(csv_path),
        "--sensitive", "gender", "--sensitive", "ethnicity",
        "--target", "hired", "--sesoi", "0.1",
        "--out", str(out), "--resamples", "200", "--seed", "7",
    ])
    assert code == 0
    assert (out / "audit.json").exists()
    assert (out / "coverage_map.png").exists()


def test_written_json_records_the_declared_sesoi(csv_path, tmp_path):
    out = tmp_path / "out"
    main([
        "audit", str(csv_path), "--sensitive", "gender", "--target", "hired",
        "--sesoi", "0.15", "--out", str(out), "--resamples", "200", "--seed", "7",
    ])
    report = json.loads((out / "audit.json").read_text())
    assert report["configuration"]["sesoi"] == pytest.approx(0.15)


def test_sesoi_is_a_required_argument(csv_path, tmp_path):
    """plan.md Sec 3.6: no silent default. The user declares what matters."""
    with pytest.raises(SystemExit) as exit_info:
        main(["audit", str(csv_path), "--sensitive", "gender",
              "--out", str(tmp_path / "out")])
    assert exit_info.value.code != 0


def test_at_least_one_sensitive_attribute_is_required(csv_path, tmp_path):
    """The tool suggests candidates; it never decides. Both review documents
    are firm on this and they are right."""
    with pytest.raises(SystemExit):
        main(["audit", str(csv_path), "--sesoi", "0.1", "--out", str(tmp_path / "o")])


def test_a_missing_file_fails_cleanly(tmp_path):
    code = main(["audit", str(tmp_path / "nope.csv"), "--sensitive", "g",
                 "--sesoi", "0.1", "--out", str(tmp_path / "out")])
    assert code == 2


def test_an_unknown_column_fails_cleanly(csv_path, tmp_path):
    code = main(["audit", str(csv_path), "--sensitive", "nonexistent",
                 "--sesoi", "0.1", "--out", str(tmp_path / "out")])
    assert code == 2


def test_demo_runs_end_to_end(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--resamples", "200"]) == 0
    assert (out / "audit.json").exists()
    assert (out / "coverage_map.png").exists()
    assert (out / "scenario.csv").exists()


def test_demo_reports_its_own_ground_truth(tmp_path):
    """The demo is only worth anything if the true effects are stated, so the
    reader can check the verdict against them."""
    out = tmp_path / "demo"
    main(["demo", "--out", str(out), "--resamples", "200"])
    truth = json.loads((out / "ground_truth.json").read_text())
    assert truth["true_effects"]["ethnicity"] > 0.1
    assert truth["true_effects"]["gender"] == 0


def test_an_audit_with_nothing_testable_does_not_crash(tmp_path):
    """Regression: plot_coverage_map raised after audit.json was written."""
    path = tmp_path / "flat.csv"
    path.write_text("g,y\n" + "a,1\n" * 20)
    out = tmp_path / "out"
    code = main(["audit", str(path), "--sensitive", "g", "--sesoi", "0.1", "--out", str(out)])
    assert code == 0
    assert (out / "audit.json").exists()


def test_a_plotting_failure_is_not_mistaken_for_an_empty_audit(csv_path, tmp_path, monkeypatch):
    """Only 'no testable cells' means no map; any other plotting error must
    surface rather than be reported as an empty audit."""
    import dbias.cli as cli

    def broken(*args, **kwargs):
        raise ValueError("Image size is too large")

    monkeypatch.setattr(cli, "plot_coverage_map", broken)
    with pytest.raises(ValueError, match="too large"):
        main([
            "audit", str(csv_path), "--sensitive", "gender", "--target", "hired",
            "--sesoi", "0.1", "--out", str(tmp_path / "out"), "--resamples", "50",
        ])


def test_a_malformed_csv_is_reported_not_raised(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff\xfe\x00bad")
    code = main(["audit", str(path), "--sensitive", "g", "--sesoi", "0.1"])
    assert code == 2
    assert "cannot read" in capsys.readouterr().err


def test_hierarchical_fdr_is_selectable_and_recorded(csv_path, tmp_path):
    out = tmp_path / "out"
    code = main([
        "audit", str(csv_path), "--sensitive", "gender", "--sensitive", "ethnicity",
        "--target", "hired", "--sesoi", "0.1", "--fdr", "hierarchical",
        "--out", str(out), "--resamples", "50", "--seed", "7",
    ])
    assert code == 0
    report = json.loads((out / "audit.json").read_text())
    assert report["configuration"]["correction"].startswith("Hierarchical")


def test_an_unknown_fdr_flag_is_an_argument_error(csv_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["audit", str(csv_path), "--sensitive", "gender", "--sesoi", "0.1", "--fdr", "global"])
    assert exit_info.value.code == 2
