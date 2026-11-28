"""Step 3: model-ready features. Check features make each data-quality error visible;
categories are one-hot encoded with the vocabulary of the CLEAN data."""
import numpy as np
import pandas as pd

from src.inject import CAT_COLS, MISSING_COLS

CHECK_COLS = [f'{c}_missing' for c in MISSING_COLS] + [
    'is_repeat', 'abs_balance_diff', 'amount_negative'] + [f'{c}_invalid' for c in CAT_COLS] + [
    'city_state_mismatch', 'hour_mismatch']


def reference(clean):
    """Rules learned from the CLEAN data: valid category values and the city -> state map."""
    return {'valid': {c: set(clean[c].unique()) for c in CAT_COLS},
            'city_state': clean.groupby('city')['state'].first().to_dict()}



def build_features(df, ref):
    """Turn a (dirty) dataset into model-ready X (all numeric, no missing values) and y."""
    d = df.copy()
    X = pd.DataFrame(index=d.index)

    # 1. missing-value flags (made BEFORE filling)
    for c in MISSING_COLS:
        X[f'{c}_missing'] = d[c].isnull().astype(int)

    # 2. duplicate check: is this an exact repeat of an earlier row?
    key = ['nameOrig', 'nameDest', 'amount', 'timestamp']
    X['is_repeat'] = (d.fillna({'amount': -999}).groupby(key, dropna=False).cumcount() > 0).astype(int)

    # 3. balance check: should be 0 for a correct row
    X['abs_balance_diff'] = (d['oldbalanceOrg'] - d['amount'] - d['newbalanceOrig']).abs().fillna(0)

    # 4. amount on a log scale, plus a negative-amount flag
    X['amount_negative'] = (d['amount'] < 0).astype(int)
    X['amount_log'] = np.log1p(d['amount'].abs())

    # 5. categorical check: value not in the valid list from the clean data
    for c in CAT_COLS:
        X[f'{c}_invalid'] = (d[c].notnull() & ~d[c].isin(ref['valid'][c])).astype(int)

    # 6. cross-field checks
    expected_state = d['city'].map(ref['city_state'])
    X['city_state_mismatch'] = (d['city'].notnull() & (expected_state != d['state'])).astype(int)
    X['hour_mismatch'] = (d['timestamp'].dt.hour != d['hour']).astype(int)

    # 7. ordinary numeric features
    for c in ['oldbalanceOrg', 'newbalanceOrig', 'hour', 'day_of_week', 'day_of_month', 'month',
              'amount_vs_average', 'customer_risk_score', 'sender_degree']:
        X[c] = d[c]

    # 8. one-hot encoding with the CLEAN vocabulary (invalid spellings get all zeros + the _invalid flag)
    for c in CAT_COLS + ['city', 'state', 'device_type', 'device_os', 'category_risk']:
        vocab = sorted(ref['valid'][c]) if c in CAT_COLS else sorted(d[c].dropna().unique())
        X = X.join(pd.get_dummies(pd.Categorical(d[c].where(d[c].isin(vocab)), categories=vocab), prefix=c).set_index(d.index).astype(int))

    # 9. remaining gaps (blanked numeric values) are left as NaN: they are filled with the median
    #    INSIDE each model pipeline, fitted on the training split only (no test-set information).
    #    The _missing flags keep the signal that a value was blank.
    return X, d['is_dq_anomaly'], d['issue_type']


def feature_sets(X):
    """Two versions of X: with the data-quality check features, and raw fields only.
    In the raw version the amount keeps its sign (a negative amount is raw data, not a check)."""
    raw = X.drop(columns=CHECK_COLS).copy()
    raw['amount_log'] = X['amount_log'] * np.where(X['amount_negative'] == 1, -1, 1)
    return {'with checks': X, 'raw only': raw}
