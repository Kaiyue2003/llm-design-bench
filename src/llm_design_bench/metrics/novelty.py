import numpy as np


def candidate_novelty(mixtures: np.ndarray, logged_mixtures: np.ndarray) -> float:
    mixtures = np.asarray(mixtures, dtype=float)
    logged_mixtures = np.asarray(logged_mixtures, dtype=float)
    if len(mixtures) == 0 or len(logged_mixtures) == 0:
        raise ValueError("mixture arrays must not be empty")
    distances = np.linalg.norm(
        mixtures[:, None, :] - logged_mixtures[None, :, :], axis=2
    )
    return float(distances.min(axis=1).mean())
