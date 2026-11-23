"""Step 4: the five detectors, fitted and evaluated the same way."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config

UNSUPERVISED = ['Isolation Forest', 'LOF', 'One-Class SVM']
SUPERVISED = ['Random Forest', 'XGBoost']


def load_features(rate, seed):
    f = pd.read_csv(config.PROCESSED / f'features_{config.dataset_tag(rate, seed)}.csv.gz')
    return f.drop(columns=['is_dq_anomaly', 'issue_type']), f['is_dq_anomaly'], f['issue_type']


def split(y, seed=config.RANDOM_STATE):
    """Stratified 70/30 split of row positions, shared by every model."""
    return train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)


def make_models(rate, y_train, params=None, seed=config.RANDOM_STATE):
    """The five models with default settings (or tuned `params` per model name).
    Unsupervised models flag the `rate` share of rows; LOF and One-Class SVM use standardised features."""
    from xgboost import XGBClassifier          # imported here so the rest of the module works without it
    params = params or {}
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    p = lambda name: params.get(name, {})
    return {
        'Isolation Forest': IsolationForest(contamination=rate, random_state=seed, n_jobs=-1, **p('Isolation Forest')),
        'LOF': make_pipeline(StandardScaler(), LocalOutlierFactor(novelty=True, contamination=rate,
                                                                  **{'n_neighbors': 20, **p('LOF')})),
        'One-Class SVM': make_pipeline(StandardScaler(), OneClassSVM(**{'nu': rate, 'gamma': 'scale', **p('One-Class SVM')})),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=-1,
                                                **{'n_estimators': 300, **p('Random Forest')}),
        'XGBoost': XGBClassifier(scale_pos_weight=neg / pos, eval_metric='logloss', random_state=seed, n_jobs=-1,
                                 **{'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 6, **p('XGBoost')}),
    }


def fit_score(name, model, X_train, y_train, X_test):
    """Fit one model; return 0/1 predictions, an anomaly score (higher = more anomalous) and seconds."""
    t = time.time()
    if name in UNSUPERVISED:
        model.fit(X_train)                              # no labels
        score = -model.decision_function(X_test)
        pred = (model.predict(X_test) == -1).astype(int)
    else:
        model.fit(X_train, y_train)
        score = model.predict_proba(X_test)[:, 1]
        pred = (score >= 0.5).astype(int)
    return pred, score, time.time() - t


def evaluate(y_true, pred, score):
    return {'precision': precision_score(y_true, pred, zero_division=0), 'recall': recall_score(y_true, pred),
            'F1': f1_score(y_true, pred), 'PR-AUC': average_precision_score(y_true, score)}


def recall_by_issue(pred, issue_test):
    """Share of test rows of each issue type that were flagged ('none' = false-alarm rate)."""
    return pd.Series(pred, index=issue_test.index).groupby(issue_test).mean()


def run_all(X_sets, y, issue, rate, params=None, seed=config.RANDOM_STATE):
    """Every model on every feature set with one shared split. Returns (results, recall_by_issue) tables."""
    idx_train, idx_test = split(y, seed)
    y_train, y_test, issue_test = y.iloc[idx_train], y.iloc[idx_test], issue.iloc[idx_test]
    rows, per_issue = [], {}
    for fs_name, X in X_sets.items():
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        for name, model in make_models(rate, y_train, params, seed).items():
            pred, score, secs = fit_score(name, model, X_train, y_train, X_test)
            rows.append({'features': fs_name, 'model': name,
                         'type': 'unsupervised' if name in UNSUPERVISED else 'supervised',
                         **evaluate(y_test, pred, score), 'seconds': secs})
            per_issue[f'{name} | {fs_name}'] = recall_by_issue(pred, issue_test)
    return pd.DataFrame(rows), pd.DataFrame(per_issue).T


def timing_test(X, sizes=(5000, 10000, 20000, 35000), full_rows=1_088_589, seed=config.RANDOM_STATE):
    """Fit time of the unsupervised models on growing samples, and a power-law extrapolation
    (time ~ rows^k) to the full dataset. X should be standardised."""
    X = np.asarray(X)
    models = {'Isolation Forest': lambda: IsolationForest(random_state=seed),
              'LOF': lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
              'One-Class SVM': lambda: OneClassSVM(nu=0.05, gamma='scale')}
    times = {name: [] for name in models}
    for n in sizes:
        for name, build in models.items():
            t = time.time(); build().fit(X[:n]); times[name].append(time.time() - t)
    times = pd.DataFrame(times, index=list(sizes))
    extrap = {}
    for name in times:
        k = np.polyfit(np.log(list(sizes)), np.log(times[name].clip(lower=1e-3)), 1)[0]
        extrap[name] = {'growth_exponent_k': k,
                        'est_minutes_full_data': times[name].iloc[-1] * (full_rows / sizes[-1]) ** k / 60}
    return times, pd.DataFrame(extrap).T
