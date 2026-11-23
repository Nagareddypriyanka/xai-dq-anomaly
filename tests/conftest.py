"""Shared test data: a small, perfectly clean transaction table built in code (no data files needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make `config` and `src` importable

CITY_STATE = {'Sydney': 'NSW', 'Melbourne': 'VIC', 'Brisbane': 'QLD', 'Perth': 'WA', 'Adelaide': 'SA'}


@pytest.fixture
def clean():
    rng = np.random.default_rng(0)
    n = 600
    city = rng.choice(list(CITY_STATE), n)
    ts = pd.Timestamp('2025-10-13') + pd.to_timedelta(rng.integers(0, 60 * 24 * 90, n), unit='min')
    old = rng.uniform(1_000, 500_000, n)
    amount = rng.uniform(10, 900, n)
    return pd.DataFrame({
        'type': rng.choice(['DEBIT', 'TRANSFER', 'BPAY'], n),
        'amount': amount,
        'category': rng.choice(['Food', 'Housing', 'Transport'], n),
        'nameOrig': [f'C{i}' for i in rng.integers(0, 200, n)],
        'nameDest': [f'C{i}' for i in rng.integers(0, 200, n)],
        'oldbalanceOrg': old,
        'newbalanceOrig': old - amount,
        'hour': ts.hour, 'day_of_week': ts.dayofweek, 'day_of_month': ts.day, 'month': ts.month,
        'timestamp': ts,
        'payment_method': rng.choice(['PayID', 'CardNumber', 'BSB_Account'], n),
        'city': city, 'state': [CITY_STATE[c] for c in city],
        'postcode': rng.integers(2000, 7000, n).astype(str),
        'device_type': rng.choice(['Mobile', 'Web', 'ATM'], n),
        'device_os': rng.choice(['iOS', 'Android'], n),
        'amount_vs_average': rng.uniform(0, 3, n),
        'customer_risk_score': rng.uniform(50, 100, n),
        'category_risk': rng.choice(['low', 'medium'], n),
        'sender_degree': rng.integers(1, 30, n),
    })
