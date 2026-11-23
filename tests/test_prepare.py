"""Metadata parsing and field screening."""
import pandas as pd

from src.prepare import parse_metadata, profile_fields, quality_checks, screen_fields

SAMPLE = ("{'timestamp': datetime.datetime(2025, 10, 13, 17, 23, 36, 636449), "
          "'location': {'city': 'Sydney', 'state': 'NSW', 'country': 'Australia'}, 'merchant_info': None}")


def test_parse_metadata_handles_datetime():
    meta = parse_metadata(pd.Series([SAMPLE, SAMPLE]))
    assert meta.loc[0, 'location.city'] == 'Sydney'
    assert meta.loc[0, 'timestamp'] == pd.Timestamp('2025-10-13 17:23:36.636449')


def test_screening_rules():
    n = 100
    meta = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='h'),
        'keep_me': ['a', 'b', 'c', 'd'] * 25,
        'constant': ['Australia'] * n,
        'empty': [None] * n,
        'near_constant': [0.0] * 95 + [1.0] * 5,
        'sparse': ['x'] * 30 + [None] * 70,
        'ip_like': [f'10.0.0.{i}' for i in range(n)]})
    reasons = screen_fields(profile_fields(meta), n, judgment={})
    assert set(reasons) == {'constant', 'empty', 'near_constant', 'sparse', 'ip_like'}


def test_quality_checks_on_clean_data(clean):
    q = quality_checks(clean)
    assert q['missing_values'] == 0 and q['duplicate_rows'] == 0 and q['balance_errors'] == 0
    assert q['hour_matches_timestamp'] == 1.0 and q['cities_in_more_than_one_state'] == 0
