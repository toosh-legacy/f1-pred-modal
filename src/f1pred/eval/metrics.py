# ============================================================================
# eval/metrics.py  —  THE SCORECARDS  (Step 13)
# ----------------------------------------------------------------------------
# After a model predicts an order, how do we grade it? These functions compare the
# PREDICTED order to the REAL finishing order and output numbers. There is NO model
# here — just the math that scores a prediction. Higher score = better prediction.
#
# Key idea: a "score" here means "higher = we think this driver finishes better".
# The real answer is `position` (1 = winner, bigger = worse, NaN = didn't finish).
# Each function grades ONE race.
# ============================================================================

"""Ranking metrics for race-result prediction.

Convention: a *score* is higher = predicted-to-finish-better. Ground truth is the
actual finishing ``position`` (1 = winner, larger = worse; NaN = did not finish).
All functions operate on a single race (one ranking group).
"""

from __future__ import annotations

import numpy as np  # numpy = fast math on arrays of numbers. "np" is the standard nickname.


# Private helper: get the order of drivers from best predicted to worst.
def _order_by_score(scores: np.ndarray) -> np.ndarray:
    """Indices that sort by score descending (best-predicted first)."""
    # argsort returns the POSITIONS that would sort the array. We negate scores so the
    # HIGHEST score comes first. "stable" keeps ties in their original order.
    return np.argsort(-scores, kind="stable")


# Convert finishing position into a "relevance" number (winner = most relevant).
# NDCG (below) needs this: it rewards putting high-relevance drivers near the top.
def relevance_from_position(position: np.ndarray, n: int | None = None) -> np.ndarray:
    """Map finishing position -> non-negative relevance (winner highest).

    NaN positions (DNF / not classified) get relevance 0. With ``n`` drivers the
    winner scores n-1, runner-up n-2, ... This is the graded relevance NDCG uses.
    """
    position = np.asarray(position, dtype=float)
    # If n not given, count how many drivers actually have a position (not NaN).
    n = int(n if n is not None else np.sum(~np.isnan(position)))
    # winner (position 1) -> n-1, second -> n-2, ... DNF (NaN) -> 0.
    rel = np.where(np.isnan(position), 0.0, np.maximum(n - position, 0.0))
    return rel.astype(float)


# DCG = "Discounted Cumulative Gain". It sums up relevance but DISCOUNTS drivers lower
# in the predicted list (getting the top ones right matters most).
def dcg_at_k(relevance_in_rank_order: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance_in_rank_order, dtype=float)[:k]  # only look at the top k
    if rel.size == 0:
        return 0.0
    # The discount shrinks the deeper you go: position 1 counts full, position 2 less, etc.
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    # (2**rel - 1) rewards higher relevance more; multiply by the discount and sum.
    return float(np.sum((2**rel - 1) * discounts))


# NDCG = "Normalized DCG": DCG divided by the BEST possible DCG, so the score is 0-1.
# 1.0 = perfect ordering. This is our main ranking-quality number.
def ndcg_at_k(scores: np.ndarray, position: np.ndarray, k: int) -> float:
    """NDCG@k for one race. Returns 0.0 for a degenerate (all-zero-relevance) race."""
    scores = np.asarray(scores, dtype=float)
    relevance = relevance_from_position(position, n=scores.size)
    order = _order_by_score(scores)  # the model's predicted order
    dcg = dcg_at_k(relevance[order], k)  # DCG of the model's order
    ideal = dcg_at_k(np.sort(relevance)[::-1], k)  # DCG of the PERFECT order
    return float(dcg / ideal) if ideal > 0 else 0.0  # normalize to 0-1


# Simple, intuitive metric: did we correctly pick the winner?
def top1_hit(scores: np.ndarray, position: np.ndarray) -> float:
    """1.0 if the top-scored driver actually won the race, else 0.0."""
    position = np.asarray(position, dtype=float)
    if np.all(np.isnan(position)):  # no one classified -> can't judge
        return 0.0
    predicted_winner = int(_order_by_score(scores)[0])  # our #1 pick (row index)
    actual_winner = int(np.nanargmin(position))  # the real winner = lowest position number
    return float(predicted_winner == actual_winner)  # 1.0 if they match


# How many of the real top-k finishers did we also put in our top-k? (order within k ignored)
def topk_overlap(scores: np.ndarray, position: np.ndarray, k: int) -> float:
    """Fraction of the actual top-k finishers captured in the predicted top-k."""
    scores = np.asarray(scores, dtype=float)
    position = np.asarray(position, dtype=float)
    pred_topk = set(_order_by_score(scores)[:k].tolist())  # our predicted top-k drivers
    # The actual order (put NaN/DNF at the very end so they don't sneak into the top).
    actual_order = np.argsort(np.where(np.isnan(position), np.inf, position), kind="stable")
    n_classified = int(np.sum(~np.isnan(position)))
    actual_topk = set(actual_order[: min(k, n_classified)].tolist())  # real top-k
    if not actual_topk:
        return 0.0
    # "&" between sets = the drivers in BOTH. Divide by how many real top-k there were.
    return len(pred_topk & actual_topk) / len(actual_topk)


# Spearman correlation: overall, does our predicted order agree with the real order?
# +1 = perfect agreement, 0 = no relationship, -1 = exactly backwards.
def spearman(scores: np.ndarray, position: np.ndarray) -> float:
    """Spearman rank correlation between predicted order and actual finish.

    Restricted to classified finishers. Returns NaN if fewer than 2 are classified.
    Positive = good (predicted-better drivers did finish better).
    """
    scores = np.asarray(scores, dtype=float)
    position = np.asarray(position, dtype=float)
    mask = ~np.isnan(position)  # only compare drivers who actually finished
    if mask.sum() < 2:  # need at least 2 to measure a correlation
        return float("nan")
    # Higher score should mean lower (better) position -> compare score vs. -position.
    a = scores[mask]
    b = -position[mask]
    ar = _rankdata(a)  # turn values into ranks (1st, 2nd, ...)
    br = _rankdata(b)
    ar = ar - ar.mean()  # center the ranks around 0
    br = br - br.mean()
    # This is the standard correlation formula on the ranks.
    denom = np.sqrt(np.sum(ar**2) * np.sum(br**2))
    return float(np.sum(ar * br) / denom) if denom > 0 else float("nan")


# Private helper: convert values into ranks, with ties sharing the average rank.
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank of values (ties share the mean rank), like scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")  # positions that sort the values
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)  # assign ranks 1..n
    # If some values are equal (ties), give them all the AVERAGE of their ranks.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)  # sum the ranks within each tied group
    return (sums / counts)[inv]  # divide by group size -> average rank
