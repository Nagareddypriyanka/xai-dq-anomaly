"""Step 1: from the raw AMLNet file to a clean, verified 50k sample."""
import ast
import re

import numpy as np
import pandas as pd

import config

# Metadata fields kept after screening get short names; device fields keep a prefix
# because 'type' would clash with the transaction type column.
RENAME_OVERRIDES = {'device_info.type': 'device_type', 'device_info.os': 'device_os'}


def load_raw(path=config.RAW_FILE):
    return pd.read_csv(path)


def remove_laundering(df):
    """Keep normal transactions only, so every anomaly later is an injected data-quality error."""
    return df[df['isMoneyLaundering'] == 0].copy()


def take_sample(df, n=config.SAMPLE_SIZE, seed=config.RANDOM_STATE):
    return df.sample(n, random_state=seed).copy()


def parse_metadata_value(text):
    """One metadata string -> dict. datetime.datetime(...) is turned into plain text first,
    then ast.literal_eval reads it safely (it accepts data only, never code)."""
    text = re.sub(r'datetime\.datetime\(([^)]*)\)', r"'\1'", text)
    return ast.literal_eval(text)


def parse_metadata(series, chunksize=100_000):
    """Whole metadata column -> one column per nested field, timestamp as a real date.
    Parsed in chunks so the full 1.09M rows fit in memory."""
    parts = []
    for start in range(0, len(series), chunksize):
        chunk = series.iloc[start:start + chunksize]
        part = pd.json_normalize(chunk.apply(parse_metadata_value).tolist())
        part.index = chunk.index
        parts.append(part)
    meta = pd.concat(parts)
    meta['timestamp'] = pd.to_datetime(meta['timestamp'], format='%Y, %m, %d, %H, %M, %S, %f')
    return meta


def profile_fields(meta):
    """Completeness, number of distinct values and dominance of the top value for every field."""
    return pd.DataFrame({
        'non_null_%': (meta.notnull().mean() * 100).round(1),
        'unique': meta.nunique(),
        'top_value_%': meta.apply(lambda c: c.value_counts(normalize=True).iloc[0] * 100
                                  if c.notnull().any() else np.nan).round(1),
        'numeric': meta.apply(pd.api.types.is_numeric_dtype),
        'example': meta.apply(lambda c: c.dropna().iloc[0] if c.notnull().any() else None)})


def screen_fields(fields, n_rows, judgment=config.JUDGMENT_DROPS):
    """Rule-based field screening. Returns {field: reason} for every field to DROP."""
    reasons = {}
    for col, r in fields.iterrows():
        if r['non_null_%'] == 0:
            reasons[col] = 'empty'
        elif r['unique'] == 1:
            reasons[col] = 'constant (1 value)'
        elif r['top_value_%'] >= config.NEAR_CONSTANT_PCT:
            reasons[col] = f"near-constant ({r['top_value_%']}% one value)"
        elif r['non_null_%'] < config.SPARSE_PCT:
            reasons[col] = f"sparse ({r['non_null_%']}% filled)"
        elif not r['numeric'] and col != 'timestamp' and r['unique'] / n_rows > config.IDENTIFIER_UNIQUE_SHARE:
            reasons[col] = f"identifier-like text ({r['unique']} unique values)"
    for col, why in judgment.items():
        if col in fields.index:
            reasons[col] = why
    return reasons


def select_and_rename(meta, reasons):
    keep = [c for c in meta.columns if c not in reasons]
    rename = {c: c.split('.')[-1] for c in keep}
    rename.update({k: v for k, v in RENAME_OVERRIDES.items() if k in keep})
    return meta[keep].rename(columns=rename)


def quality_checks(data):
    """Evidence that the baseline is clean. Every value should be 0 (or 1.0 for the matches)."""
    diff = data['oldbalanceOrg'] - data['amount'] - data['newbalanceOrig']
    return {
        'missing_values': int(data.isnull().sum().sum()),
        'duplicate_rows': int(data.duplicated().sum()),
        'balance_errors': int((diff.abs() > 0.01).sum()),
        'negative_or_zero_amounts': int((data['amount'] <= 0).sum()),
        'negative_balances': int((data['newbalanceOrig'] < 0).sum()),
        'hour_matches_timestamp': float((data['timestamp'].dt.hour == data['hour']).mean()),
        'cities_in_more_than_one_state': int((data.groupby('city')['state'].nunique() > 1).sum()),
    }


def representativeness(full, sample, col):
    """Share of each value of `col` in the full data vs the sample (percent)."""
    return pd.DataFrame({'full_%': full[col].value_counts(normalize=True) * 100,
                         'sample_%': sample[col].value_counts(normalize=True) * 100}).round(2)


def load_clean(path=config.CLEAN_FILE):
    """Load the saved clean sample with the right data types."""
    df = pd.read_csv(path, dtype={'postcode': str})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def clean_table(normal):
    """Metadata parsing, rule-based field screening and column removal for any set of normal rows
    (the 50k sample or the full data). Returns the clean table and the field-screening table."""
    meta = parse_metadata(normal['metadata'])
    fields = profile_fields(meta)
    reasons = screen_fields(fields, len(meta))
    fields['decision'] = ['DROP: ' + reasons[c] if c in reasons else 'keep' for c in fields.index]
    data = pd.concat([normal, select_and_rename(meta, reasons)], axis=1)
    data = data.drop(columns=config.LEAKAGE_COLS + config.USELESS_COLS)
    assert not data.columns.duplicated().any()
    return data, fields


def prepare(raw_path=config.RAW_FILE, out_path=config.CLEAN_FILE):
    """Full step 1. Returns the clean sample and the field-screening table; saves the sample."""
    data, fields = clean_table(take_sample(remove_laundering(load_raw(raw_path))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    return data, fields

def laundering_rows(raw):
    """The money-laundering transactions removed from the main experiment (used only for the false-alarm check)."""
    return raw[raw['isMoneyLaundering'] == 1].copy()


def clean_like(rows, fields):
    """Clean extra rows (e.g. the laundering transactions) with EXACTLY the field decisions already made on the
    normal data (`fields` from clean_table), so they get the same columns. No new screening is done."""
    meta = parse_metadata(rows['metadata'])
    keep = list(fields.index[fields['decision'] == 'keep'])
    for c in keep:                      # a kept field absent from these rows becomes an empty column
        if c not in meta.columns:
            meta[c] = np.nan
    rename = {c: c.split('.')[-1] for c in keep}
    rename.update({k: v for k, v in RENAME_OVERRIDES.items() if k in keep})
    data = pd.concat([rows, meta[keep].rename(columns=rename)], axis=1)
    return data.drop(columns=[c for c in config.LEAKAGE_COLS + config.USELESS_COLS if c in data.columns])
