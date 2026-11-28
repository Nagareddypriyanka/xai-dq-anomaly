"""Each check feature must fire on its own issue and never on clean rows."""
import pandas as pd

from src.features import CHECK_COLS, build_features, feature_sets, reference
from src.inject import make_dirty

CHECK_FOR_ISSUE = {
    'missing': lambda X: X.filter(like='_missing').sum(axis=1) > 0,
    'duplicate': lambda X: X['is_repeat'] == 1,
    'balance': lambda X: X['abs_balance_diff'] > 0.01,
    'numerical': lambda X: X['abs_balance_diff'] > 0.01,
    'categorical': lambda X: X.filter(like='_invalid').sum(axis=1) > 0,
    'crossfield': lambda X: (X['city_state_mismatch'] + X['hour_mismatch']) > 0,
}


def test_clean_data_fires_no_checks(clean):
    X, y, _ = build_features(clean.assign(is_dq_anomaly=0, issue_type='none'), reference(clean))
    assert X[CHECK_COLS].drop(columns='abs_balance_diff').values.sum() == 0
    assert (X['abs_balance_diff'] > 0.01).sum() == 0


def test_each_issue_fires_its_check(clean):
    dirty = make_dirty(clean, rate=0.3, seed=2)
    X, _, issue = build_features(dirty, reference(clean))
    for name, check in CHECK_FOR_ISSUE.items():
        assert check(X)[issue == name].all(), f'{name} not detected by its check'


def test_features_numeric_and_gaps_only_where_blanked(clean):
    """Blanked values stay NaN (imputed later inside each model pipeline); every NaN row is flagged as missing."""
    X, _, issue = build_features(make_dirty(clean, rate=0.3, seed=3), reference(clean))
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    has_gap = X.isnull().any(axis=1)
    assert (issue[has_gap] == 'missing').all()
    assert (X.loc[has_gap].filter(like='_missing').sum(axis=1) > 0).all()


def test_invalid_spelling_gets_no_onehot_column(clean):
    dirty = make_dirty(clean, rate=0.3, seed=4)
    X, _, _ = build_features(dirty, reference(clean))
    bad = X['type_invalid'] == 1
    assert bad.any()
    assert (X.loc[bad, X.columns.str.startswith('type_') & ~X.columns.str.endswith('invalid')].sum(axis=1) == 0).all()


def test_raw_feature_set_has_no_checks(clean):
    X, _, _ = build_features(make_dirty(clean, rate=0.3, seed=5), reference(clean))
    sets = feature_sets(X)
    assert not set(CHECK_COLS) & set(sets['raw only'].columns)
    assert set(CHECK_COLS) <= set(sets['with checks'].columns)
