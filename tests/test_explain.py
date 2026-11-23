import pandas as pd
import pytest

from src.explain import rank_spearman, top_feature_matches, topk_jaccard


def test_jaccard_and_spearman():
    a = pd.Series([5, 4, 3, 2, 1], index=list('abcde'))
    assert topk_jaccard(a, a, k=3) == 1.0
    assert rank_spearman(a, a) == pytest.approx(1.0)
    b = pd.Series([1, 2, 3, 4, 5], index=list('abcde'))
    assert topk_jaccard(a, b, k=2) == 0.0
    assert rank_spearman(a, b) == pytest.approx(-1.0)


def test_top_feature_matches():
    assert top_feature_matches('abs_balance_diff', 'balance')
    assert top_feature_matches('type_invalid', 'categorical')
    assert not top_feature_matches('category_missing', 'categorical')
    assert not top_feature_matches('hour', 'balance')
