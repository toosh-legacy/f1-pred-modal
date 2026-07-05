import numpy as np

from f1pred.eval import metrics


def test_perfect_ranking_scores_one():
    # scores descending match positions ascending -> perfect.
    position = np.array([1, 2, 3, 4, 5], dtype=float)
    scores = np.array([5, 4, 3, 2, 1], dtype=float)
    assert metrics.ndcg_at_k(scores, position, 3) == 1.0
    assert metrics.top1_hit(scores, position) == 1.0
    assert metrics.topk_overlap(scores, position, 3) == 1.0
    assert metrics.spearman(scores, position) == 1.0


def test_reversed_ranking_is_bad():
    position = np.array([1, 2, 3, 4, 5], dtype=float)
    scores = np.array([1, 2, 3, 4, 5], dtype=float)  # winner scored lowest
    assert metrics.top1_hit(scores, position) == 0.0
    assert metrics.spearman(scores, position) == -1.0
    assert metrics.ndcg_at_k(scores, position, 3) < 1.0


def test_dnf_positions_are_ignored_as_relevance():
    position = np.array([1, 2, np.nan, np.nan], dtype=float)
    scores = np.array([4, 3, 2, 1], dtype=float)
    # Two DNFs contribute zero relevance; top-2 still perfectly ordered.
    assert metrics.ndcg_at_k(scores, position, 2) == 1.0
    assert metrics.topk_overlap(scores, position, 3) == 1.0  # only 2 classified


def test_topk_overlap_partial():
    position = np.array([1, 2, 3, 4], dtype=float)
    # Predict 1 and 3 in top-2; actual top-2 is {0,1} -> overlap 1/2.
    scores = np.array([5, 1, 4, 0], dtype=float)
    assert metrics.topk_overlap(scores, position, 2) == 0.5


def test_rankdata_handles_ties():
    r = metrics._rankdata(np.array([10.0, 10.0, 20.0]))
    assert np.allclose(r, [1.5, 1.5, 3.0])
