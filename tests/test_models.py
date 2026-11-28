"""Split, imputation inside the pipeline, and the extra-rows (laundering) check."""
import numpy as np
import pandas as pd

from src.features import build_features, feature_sets, reference
from src.inject import make_dirty
from src.models import run_all, split


def test_split_keeps_every_issue_type_70_30(clean):
    dirty = make_dirty(clean, rate=0.3, seed=6)
    issue = dirty['issue_type']
    tr, te = split(issue)
    share = issue.iloc[te].value_counts() / issue.value_counts()
    assert ((share - 0.3).abs() < 0.03).all()


def test_run_all_handles_blanks_and_reports_extra_flag_rate(clean):
    dirty = make_dirty(clean, rate=0.3, seed=7)
    X, y, issue = build_features(dirty, reference(clean))
    assert X.isnull().any().any()                      # blanks reach the models ...
    extra_rows = clean.sample(40, random_state=0).assign(is_dq_anomaly=0, issue_type='none')
    Xe, _, _ = build_features(extra_rows, reference(clean))
    res, _ = run_all(feature_sets(X), y, issue, y.mean(), only=['Isolation Forest', 'Random Forest'],
                     extra=feature_sets(Xe))
    assert len(res) == 4                               # ... and are imputed inside each pipeline
    assert res['extra_flag_rate'].between(0, 1).all()
