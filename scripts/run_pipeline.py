"""Run the project end to end from the command line.

    python scripts/run_pipeline.py              # all steps
    python scripts/run_pipeline.py --steps inject baseline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.features import build_features, feature_sets, reference
from src.inject import make_dirty
from src.models import load_features, run_all
from src.prepare import load_clean, prepare, quality_checks


def step_prepare():
    data, fields = prepare()
    print('Clean sample:', data.shape, '| metadata fields kept:', (fields['decision'] == 'keep').sum(), 'of', len(fields))
    print('Quality checks:', quality_checks(data))
    config.RESULTS.mkdir(exist_ok=True)
    fields.drop(columns='example').to_csv(config.RESULTS / 'metadata_field_screening.csv')


def step_inject():
    clean = load_clean()
    ref = reference(clean)
    for rate in config.RATES:
        for seed in config.SEEDS:
            tag = config.dataset_tag(rate, seed)
            dirty = make_dirty(clean, rate, seed)
            X, y, issue = build_features(dirty, ref)
            dirty.to_csv(config.PROCESSED / f'dirty_{tag}.csv.gz', index=False)
            X.assign(is_dq_anomaly=y, issue_type=issue).to_csv(config.PROCESSED / f'features_{tag}.csv.gz', index=False)
            print(f'{tag}: {len(dirty)} rows, {int(y.sum())} anomalies')


def step_baseline(rate=0.05, seed=42):
    X, y, issue = load_features(rate, seed)
    results, per_issue = run_all(feature_sets(X), y, issue, y.mean())
    results.to_csv(config.RESULTS / 'baseline_results.csv', index=False)
    per_issue.to_csv(config.RESULTS / 'baseline_recall_by_issue.csv')
    print(results.round(3).to_string(index=False))


STEPS = {'prepare': step_prepare, 'inject': step_inject, 'baseline': step_baseline}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', nargs='+', choices=list(STEPS), default=list(STEPS))
    for step in parser.parse_args().steps:
        print(f'===== {step}')
        STEPS[step]()
