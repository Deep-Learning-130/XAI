"""Shared machinery for the evaluation protocol (docs/06).

These are not unit tests. They measure the tool's detection accuracy against
data whose true effect is known by construction, which makes them the only
tests in the project that can fail for a *scientific* reason rather than a
programming one. They are slower than the unit suite and are meant to be run
and read on their own:

    pytest tests/synthetic -v
"""
import pandas as pd
import pytest

from dbias.audit import audit

SESOI = 0.1


def audit_frame(df: pd.DataFrame, sensitive="group", target=None, **kwargs):
    defaults = dict(sesoi=SESOI, seed=7, n_resamples=600)
    defaults.update(kwargs)
    return audit(df, sensitive_cols=[sensitive], target_col=target, **defaults)


def missingness_finding(result, feature="value", attribute="group"):
    ident = f"MAR_{feature.upper()}_{attribute.upper()}"
    matches = [f for f in result.findings if f.id == ident]
    if not matches:
        pytest.fail(f"no finding {ident} in {[f.id for f in result.findings]}")
    return matches[0]
