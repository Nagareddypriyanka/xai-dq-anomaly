"""Step 5: explanation stability measures. SHAP and LIME wrappers are added in notebook 05."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def global_importance(attributions, feature_names):
    """Mean absolute attribution per feature (from SHAP or LIME), sorted high to low."""
    return pd.Series(np.abs(np.asarray(attributions)).mean(axis=0), index=feature_names).sort_values(ascending=False)


def topk_jaccard(imp_a, imp_b, k=5):
    """Overlap of the top-k features of two importance rankings: 1 = same top-k, 0 = none shared."""
    a, b = set(imp_a.sort_values(ascending=False).index[:k]), set(imp_b.sort_values(ascending=False).index[:k])
    return len(a & b) / len(a | b)


def rank_spearman(imp_a, imp_b):
    """Spearman rank correlation of two importance vectors over the same features: 1 = identical ranking."""
    common = imp_a.index.intersection(imp_b.index)
    return spearmanr(imp_a[common], imp_b[common]).statistic


def top_feature_matches(top_feature, issue, expected=None):
    """RQ3 check: does the most important feature for a flagged row point to its injected issue?"""
    expected = expected or EXPECTED_FEATURES
    if issue != 'missing' and top_feature.endswith('_missing'):
        return False
    return any(top_feature.startswith(prefix) for prefix in expected.get(issue, []))


# Which features count as 'pointing to the right cause' for each injected issue
EXPECTED_FEATURES = {
    'missing': ['amount_missing', 'oldbalanceOrg_missing', 'newbalanceOrig_missing', 'category_missing', 'city_missing'],
    'duplicate': ['is_repeat'],
    'balance': ['abs_balance_diff', 'newbalanceOrig'],
    'numerical': ['amount_log', 'amount_negative', 'abs_balance_diff'],
    'categorical': ['type_', 'category_', 'payment_method_'],
    'crossfield': ['city_state_mismatch', 'hour_mismatch', 'state_', 'hour'],
}
