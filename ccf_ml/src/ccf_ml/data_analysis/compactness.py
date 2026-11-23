"""Within-cell quantile maps, signed gaps, and compactness lenses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances_chunked

from ccf_ml.data_analysis.constants import (
    DOUBLE_ABS,
    JOINT_RADIUS,
    K_NEIGHBORS,
    PAIR_CAP,
)


def midrank_quantiles(values: np.ndarray) -> np.ndarray:
    """Map values to mid-ranks in ``(0, 1]``: average rank divided by ``n``."""
    if values.ndim != 1:
        raise ValueError(f"values must be 1-d, got ndim={values.ndim}")
    n = int(values.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    ranks = rankdata(values, method="average")
    return ranks.astype(np.float64) / float(n)


def _k_nearest_abs_mean(
    query: np.ndarray,
    *,
    reference_sorted: np.ndarray,
    k: int,
) -> np.ndarray:
    """Mean absolute gap to the ``k`` nearest reference points on the line."""
    n_ref = int(reference_sorted.size)
    n_query = int(query.size)
    k_use = min(k, n_ref)
    out = np.empty(n_query, dtype=np.float64)
    for i in range(n_query):
        qi = float(query[i])
        pos = int(np.searchsorted(reference_sorted, qi, side="left"))
        left = pos - 1
        right = pos
        taken = 0
        acc = 0.0
        while taken < k_use:
            if left < 0:
                acc += abs(float(reference_sorted[right]) - qi)
                right += 1
                taken += 1
                continue
            if right >= n_ref:
                acc += abs(float(reference_sorted[left]) - qi)
                left -= 1
                taken += 1
                continue
            d_left = abs(float(reference_sorted[left]) - qi)
            d_right = abs(float(reference_sorted[right]) - qi)
            if d_left <= d_right:
                acc += d_left
                left -= 1
            else:
                acc += d_right
                right += 1
            taken += 1
        out[i] = acc / float(k_use)
    return out


def _k_farthest_abs_mean(
    query: np.ndarray,
    *,
    reference_sorted: np.ndarray,
    k: int,
) -> np.ndarray:
    """Mean absolute gap to the ``k`` farthest reference points on the line."""
    n_ref = int(reference_sorted.size)
    n_query = int(query.size)
    k_use = min(k, n_ref)
    out = np.empty(n_query, dtype=np.float64)
    for i in range(n_query):
        qi = float(query[i])
        left = 0
        right = n_ref - 1
        taken = 0
        acc = 0.0
        while taken < k_use:
            d_left = abs(float(reference_sorted[left]) - qi)
            d_right = abs(float(reference_sorted[right]) - qi)
            if d_left >= d_right:
                acc += d_left
                left += 1
            else:
                acc += d_right
                right -= 1
            taken += 1
        out[i] = acc / float(k_use)
    return out


def _count_within_abs(
    query: np.ndarray,
    *,
    reference_sorted: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Count reference points with ``|q_ref - q_query| <= radius``."""
    lo = np.searchsorted(reference_sorted, query - radius, side="left")
    hi = np.searchsorted(reference_sorted, query + radius, side="right")
    return (hi - lo).astype(np.int64)


def compactness_1d(
    *,
    values: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    k: int = K_NEIGHBORS,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Per-fraud compactness on one numeric, plus cell rollup.

    ``values`` / ``y`` / ``ids`` are the rows of one profile cell. ``y`` is
    binary. Quantiles use the pooled cell mid-rank CDF.
    """
    q = midrank_quantiles(values.astype(np.float64))
    fraud = y.astype(np.int64) == 1
    q1 = q[fraud]
    q0 = q[~fraud]
    ids1 = ids[fraud]
    ids0 = ids[~fraud]
    n_fraud = int(q1.size)
    n_nonfraud = int(q0.size)
    q0_sorted = np.sort(q0)
    q0_min = float(q0_sorted[0])
    q0_max = float(q0_sorted[-1])
    min_d = q1 - q0_max
    max_d = q1 - q0_min
    nn_mean = _k_nearest_abs_mean(q1, reference_sorted=q0_sorted, k=k)
    far_mean = _k_farthest_abs_mean(q1, reference_sorted=q0_sorted, k=k)
    doubles_tight = _count_within_abs(
        q1, reference_sorted=q0_sorted, radius=DOUBLE_ABS[0]
    )
    doubles_loose = _count_within_abs(
        q1, reference_sorted=q0_sorted, radius=DOUBLE_ABS[1]
    )
    mean_d = float(q1.mean() - q0.mean())
    per_fraud = {
        "id": ids1.astype(np.int64, copy=False),
        "q": q1,
        "min_d": min_d,
        "max_d": max_d,
        "nn5_abs_mean": nn_mean,
        "far5_abs_mean": far_mean,
        "n_doubles_0_01": doubles_tight,
        "n_doubles_0_05": doubles_loose,
    }
    rollup = {
        "n_fraud": float(n_fraud),
        "n_nonfraud": float(n_nonfraud),
        "mean_d": mean_d,
        "auc": 0.5 * (mean_d + 1.0),
        "min_d_global": float(min_d.min()),
        "max_d_global": float(max_d.max()),
        "median_nn5": float(np.median(nn_mean)),
        "share_almost_double": float(np.mean(doubles_tight > 0)),
        "share_separated": float(np.mean(min_d > 0.0)),
    }
    extras = {
        "q1": q1,
        "q0": q0,
        "ids1": ids1.astype(np.int64, copy=False),
        "ids0": ids0.astype(np.int64, copy=False),
    }
    return per_fraud, rollup, extras


def maybe_outer_difference(
    q1: np.ndarray,
    q0: np.ndarray,
    *,
    pair_cap: int = PAIR_CAP,
) -> np.ndarray | None:
    """Explicit ``float32`` outer difference, or ``None`` if above ``pair_cap``."""
    n_pairs = int(q1.size) * int(q0.size)
    if n_pairs == 0 or n_pairs > pair_cap:
        return None
    return (q1[:, None] - q0[None, :]).astype(np.float32)


def compactness_joint(
    *,
    quantile_matrix: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    k: int = K_NEIGHBORS,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Per-fraud Euclidean compactness on a within-cell quantile vector.

    ``quantile_matrix`` has one row per cell row and one column per coordinate.
    """
    if quantile_matrix.ndim != 2:
        raise ValueError("quantile_matrix must be 2-d")
    fraud = y.astype(np.int64) == 1
    q1 = quantile_matrix[fraud]
    q0 = quantile_matrix[~fraud]
    ids1 = ids[fraud]
    n_fraud = int(q1.shape[0])
    n_nonfraud = int(q0.shape[0])
    k_use = min(k, n_nonfraud)
    nn = NearestNeighbors(n_neighbors=k_use, algorithm="kd_tree")
    nn.fit(q0)
    nn_dist, _nn_idx = nn.kneighbors(q1)
    nn_mean = nn_dist.mean(axis=1)
    far_mean = _k_farthest_euclidean_mean(q1, q0, k=k_use)
    n_r0 = _count_within_euclidean(q1, q0, radius=JOINT_RADIUS[0])
    n_r1 = _count_within_euclidean(q1, q0, radius=JOINT_RADIUS[1])
    per_fraud = {
        "id": ids1.astype(np.int64, copy=False),
        "nn5_abs_mean": nn_mean,
        "far5_abs_mean": far_mean,
        "n_doubles_0_05": n_r0,
        "n_doubles_0_10": n_r1,
    }
    rollup = {
        "n_fraud": float(n_fraud),
        "n_nonfraud": float(n_nonfraud),
        "median_nn5": float(np.median(nn_mean)),
        "share_almost_double": float(np.mean(n_r0 > 0)),
        "share_separated": float("nan"),
        "mean_d": float("nan"),
        "auc": float("nan"),
        "min_d_global": float("nan"),
        "max_d_global": float("nan"),
    }
    extras = {"q1": q1, "q0": q0, "ids1": ids1.astype(np.int64, copy=False)}
    extras["ids0"] = ids[~fraud].astype(np.int64, copy=False)
    return per_fraud, rollup, extras


def _k_farthest_euclidean_mean(
    q1: np.ndarray,
    q0: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Mean of the ``k`` largest Euclidean distances from each query to ``q0``."""
    n_query = int(q1.shape[0])
    out = np.empty(n_query, dtype=np.float64)
    offset = 0
    n_ref = int(q0.shape[0])
    k_use = min(k, n_ref)
    for chunk in pairwise_distances_chunked(q1, q0):
        n_chunk = int(chunk.shape[0])
        if n_ref <= k_use:
            out[offset : offset + n_chunk] = chunk.mean(axis=1)
        else:
            partitioned = np.partition(chunk, kth=n_ref - k_use, axis=1)
            out[offset : offset + n_chunk] = partitioned[:, -k_use:].mean(axis=1)
        offset += n_chunk
    return out


def _count_within_euclidean(
    q1: np.ndarray,
    q0: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Count ``q0`` points within Euclidean ``radius`` of each ``q1`` row."""
    nn = NearestNeighbors(radius=radius, algorithm="kd_tree")
    nn.fit(q0)
    neighbors = nn.radius_neighbors(q1, return_distance=False)
    n_query = int(q1.shape[0])
    counts = np.empty(n_query, dtype=np.int64)
    for i in range(n_query):
        counts[i] = int(np.asarray(neighbors[i]).size)
    return counts


def write_pair_arrays(
    directory: Path,
    *,
    extras: dict[str, np.ndarray],
    pair_cap: int = PAIR_CAP,
    write_d_ij: bool = True,
) -> None:
    """Write ``q`` / id vectors. Explicit ``d_ij`` only if ``write_d_ij`` and under cap.

    Window cells keep the vectors (the signed 1-d matrix is rank-1 in them) and
    skip the dense array so overlapping hour/age families do not fill the disk.
    """
    directory.mkdir(parents=True, exist_ok=True)
    q1 = extras["q1"]
    q0 = extras["q0"]
    np.save(directory / "q1.npy", q1)
    np.save(directory / "q0.npy", q0)
    np.save(directory / "ids1.npy", extras["ids1"])
    np.save(directory / "ids0.npy", extras["ids0"])
    if not write_d_ij:
        return
    if q1.ndim == 1:
        d_ij = maybe_outer_difference(q1, q0, pair_cap=pair_cap)
        if d_ij is not None:
            np.save(directory / "d_ij.npy", d_ij)
        return
    n_pairs = int(q1.shape[0]) * int(q0.shape[0])
    if n_pairs == 0 or n_pairs > pair_cap:
        return
    d_ij = np.empty((q1.shape[0], q0.shape[0]), dtype=np.float32)
    row = 0
    n_query = int(q1.shape[0])
    while row < n_query:
        stop = min(row + 256, n_query)
        diff = q1[row:stop, None, :] - q0[None, :, :]
        d_ij[row:stop] = np.sqrt((diff * diff).sum(axis=2)).astype(np.float32)
        row = stop
    np.save(directory / "d_ij.npy", d_ij)
