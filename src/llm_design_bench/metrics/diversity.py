import numpy as np


def pairwise_diversity(mixtures: np.ndarray) -> float:
    mixtures = np.asarray(mixtures, dtype=float)
    if len(mixtures) < 2:
        return 0.0
    distances = np.linalg.norm(mixtures[:, None, :] - mixtures[None, :, :], axis=2)
    upper_triangle = distances[np.triu_indices(len(mixtures), k=1)]
    return float(upper_triangle.mean())


def mixture_entropy(mixtures: np.ndarray) -> np.ndarray:
    mixtures = np.asarray(mixtures, dtype=float)
    safe_mixtures = np.where(mixtures > 0, mixtures, 1.0)
    return -np.sum(np.where(mixtures > 0, mixtures * np.log(safe_mixtures), 0.0), axis=1)


def active_domain_count(mixtures: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    mixtures = np.asarray(mixtures, dtype=float)
    return np.sum(mixtures > threshold, axis=1)


def mixture_diagnostics(mixtures: np.ndarray) -> dict[str, float]:
    entropy = mixture_entropy(mixtures)
    active = active_domain_count(mixtures)
    return {
        "candidate_diversity": pairwise_diversity(mixtures),
        "mean_mixture_entropy": float(entropy.mean()),
        "mean_active_domain_count": float(active.mean()),
    }
