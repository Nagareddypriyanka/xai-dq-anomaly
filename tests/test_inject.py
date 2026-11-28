"""The injection must corrupt exactly the planned rows, one issue per row, reproducibly."""
import numpy as np

from src.inject import ISSUES, make_dirty


def test_counts_per_issue(clean):
    dirty = make_dirty(clean, rate=0.3, seed=1)
    per_issue = round(0.3 * len(clean) / 6)
    counts = dirty['issue_type'].value_counts()
    for issue in ISSUES:
        assert counts[issue] == per_issue
    assert dirty['is_dq_anomaly'].sum() == 6 * per_issue
    assert len(dirty) == len(clean) + per_issue            # duplicates are appended rows


def test_untouched_rows_are_unchanged(clean):
    dirty = make_dirty(clean, rate=0.3, seed=1)
    original = dirty.iloc[:len(clean)]
    untouched = original['issue_type'] == 'none'
    cols = list(clean.columns)
    assert original.loc[untouched, cols].equals(clean.loc[untouched.values, cols])


def test_duplicates_are_exact_copies_of_clean_rows(clean):
    dirty = make_dirty(clean, rate=0.3, seed=1)
    dups = dirty[dirty['issue_type'] == 'duplicate'][list(clean.columns)]
    merged = dups.merge(clean, how='left', indicator=True)
    assert (merged['_merge'] == 'both').all()


def test_same_seed_same_result_different_seed_differs(clean):
    a, b, c = make_dirty(clean, 0.3, 5), make_dirty(clean, 0.3, 5), make_dirty(clean, 0.3, 6)
    assert a.equals(b)
    assert not a['issue_type'].equals(c['issue_type'])


def test_numerical_errors_change_only_amount(clean):
    dirty = make_dirty(clean, rate=0.3, seed=1)
    rows = dirty.index[dirty['issue_type'] == 'numerical']
    ratio = dirty.loc[rows, 'amount'] / clean.loc[rows, 'amount']
    assert set(np.round(ratio, 6)) <= {10.0, 100.0, -1.0}
    assert dirty.loc[rows, 'newbalanceOrig'].equals(clean.loc[rows, 'newbalanceOrig'])


def test_categorical_variants_are_never_valid_and_include_typos(clean):
    import numpy as np
    from src.inject import _swap, _variant
    rng = np.random.default_rng(0)
    assert _swap('Housing', np.random.default_rng(1)) != 'Housing'
    assert sorted(_swap('Housing', np.random.default_rng(1))) == sorted('Housing')    # same letters, swapped
    for v in ['Food', 'TRANSFER', 'BPAY', 'PayID']:
        for _ in range(50):
            assert _variant(v, rng) != v
