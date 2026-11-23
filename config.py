"""Project settings: one place for paths, sample size, severities, seeds and screening thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / 'data' / 'raw' / 'AMLNet_v2_transactions.csv'
PROCESSED = ROOT / 'data' / 'processed'
CLEAN_FILE = PROCESSED / 'aml_clean_sample.csv'
RESULTS = ROOT / 'results'

SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

RATES = [0.01, 0.05, 0.10]          # share of rows corrupted (severity levels)
SEEDS = [42, 7, 123]                # three independent corruptions per severity

# Metadata field screening thresholds (rules of thumb, stated in the methodology)
NEAR_CONSTANT_PCT = 90              # drop if one value covers >= 90% of rows
SPARSE_PCT = 50                     # drop if < 50% of rows are filled
IDENTIFIER_UNIQUE_SHARE = 0.5       # drop text fields with > 50% unique values
JUDGMENT_DROPS = {'risk_indicators.risk_score': 'undocumented composite score (judgment)'}

# Columns removed from the transaction table
LEAKAGE_COLS = ['isMoneyLaundering', 'laundering_typology', 'fraud_probability']
USELESS_COLS = ['metadata', 'step', 'isFraud']


def dataset_tag(rate, seed):
    """File tag for a dirty dataset, e.g. rate 0.05 and seed 42 -> 'r05_s42'."""
    return f'r{int(round(rate * 100)):02d}_s{seed}'
