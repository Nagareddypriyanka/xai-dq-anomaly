"""Step 2: inject six types of data-quality error into the clean sample, with an exact answer key."""
import numpy as np
import pandas as pd

ISSUES = ['missing', 'duplicate', 'balance', 'numerical', 'categorical', 'crossfield']
CAT_COLS = ['type', 'category', 'payment_method']          # columns that receive categorical errors
MISSING_COLS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'category', 'city']


# ---------- the six injection functions: each corrupts the given rows of df in place ----------

def inject_missing(df, rows, rng):
    """Blank one important field per row."""
    for r in rows:
        df.loc[r, rng.choice(MISSING_COLS)] = np.nan


def inject_balance(df, rows, rng):
    """Wrong closing balance: old - amount no longer equals new."""
    offset = rng.uniform(10, 1000, len(rows)) * rng.choice([-1, 1], len(rows))
    df.loc[rows, 'newbalanceOrig'] = df.loc[rows, 'newbalanceOrig'] + offset


def inject_numerical(df, rows, rng):
    """Amount entered wrongly: decimal slip (x10, x100) or a wrong sign; balances keep the true values."""
    for r in rows:
        kind = rng.choice(['x10', 'x100', 'negative'])
        df.loc[r, 'amount'] = df.loc[r, 'amount'] * {'x10': 10, 'x100': 100, 'negative': -1}[kind]


def _variant(v, rng):
    """A spelling variant of a category value that is not in the valid list."""
    options = [v.lower(), v.upper(), v + ' ', ' ' + v, v[:4]]
    options = [o for o in options if o != v]
    return rng.choice(options)


def inject_categorical(df, rows, rng):
    """Inconsistent category labels: 'TRANSFER' -> 'transfer', 'Food' -> 'FOOD', 'Housing' -> 'Hous', ..."""
    for r in rows:
        col = rng.choice(CAT_COLS)
        df.loc[r, col] = _variant(df.loc[r, col], rng)


def inject_crossfield(df, rows, rng):
    """Fields that contradict each other: city in the wrong state, or hour not matching the timestamp."""
    states = sorted(df['state'].dropna().unique())
    for r in rows:
        if rng.random() < 0.5:
            df.loc[r, 'state'] = rng.choice([s for s in states if s != df.loc[r, 'state']])
        else:
            df.loc[r, 'hour'] = (df.loc[r, 'hour'] + rng.integers(1, 24)) % 24


def make_dirty(clean, rate, seed):
    """Corrupt `rate` (e.g. 0.05) of the rows, split evenly over the 6 issue types.
    Adds the answer key: is_dq_anomaly (0/1) and issue_type."""
    rng = np.random.default_rng(seed)
    df = clean.copy().reset_index(drop=True)
    df['is_dq_anomaly'] = 0
    df['issue_type'] = 'none'
    per_issue = int(round(rate * len(clean) / len(ISSUES)))

    # pick distinct rows for the 5 in-place issues
    chosen = rng.choice(len(df), size=per_issue * 5, replace=False)
    groups = dict(zip(['missing', 'balance', 'numerical', 'categorical', 'crossfield'], np.split(chosen, 5)))
    funcs = {'missing': inject_missing, 'balance': inject_balance, 'numerical': inject_numerical,
             'categorical': inject_categorical, 'crossfield': inject_crossfield}
    for issue, rows in groups.items():
        funcs[issue](df, rows, rng)
        df.loc[rows, ['is_dq_anomaly', 'issue_type']] = [1, issue]

    # duplicates: copy untouched rows and append the copies (only the copy is the error)
    untouched = np.setdiff1d(np.arange(len(df)), chosen)
    src = rng.choice(untouched, size=per_issue, replace=False)
    dups = df.loc[src].copy()
    dups[['is_dq_anomaly', 'issue_type']] = [1, 'duplicate']
    df = pd.concat([df, dups], ignore_index=True)
    return df


