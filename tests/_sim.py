"""Small simulation designs for the tests: S1_fixed and S3 of the gg-BFA paper.

S1_fixed: five communities of 30, 26, 22, 18 and 14 variables with unit loadings,
equicorrelated factors (correlation 0.3), within-community correlations 0.6 to 0.35,
and 90 background variables (p = 200).  S3: the same with primary loadings drawn
once from U(0.5, 1.5).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

SIZES = (30, 26, 22, 18, 14)
CORRELATIONS = (0.6, 0.5, 0.45, 0.4, 0.35)
P = 200
S3_LOADING_SEED = 2610137003


@dataclass(frozen=True)
class Spec:
    p: int
    blocks: tuple
    phi: np.ndarray
    residual: np.ndarray
    loadings: Optional[np.ndarray] = None


def make_spec(name):
    starts = np.r_[0, np.cumsum(SIZES)]
    blocks = tuple(tuple(range(starts[h], starts[h + 1])) for h in range(len(SIZES)))
    phi = np.full((5, 5), 0.3) + 0.7 * np.eye(5)
    residual = np.array([(1 - r) / r for r in CORRELATIONS])
    if name == "S1_fixed":
        return Spec(P, blocks, phi, residual)
    if name == "S3":
        loadings = np.zeros(P)
        loadings[: starts[-1]] = np.random.default_rng(S3_LOADING_SEED).uniform(0.5, 1.5, size=starts[-1])
        return Spec(P, blocks, phi, residual, loadings)
    raise ValueError(name)


def covariance(spec):
    lam = np.zeros((spec.p, len(spec.blocks)))
    d = np.ones(spec.p)
    for h, block in enumerate(spec.blocks):
        lam[list(block), h] = 1.0 if spec.loadings is None else spec.loadings[list(block)]
        d[list(block)] = spec.residual[h]
    return lam @ spec.phi @ lam.T + np.diag(d)


def correlation(spec):
    omega = covariance(spec)
    scale = 1 / np.sqrt(np.diag(omega))
    return scale[:, None] * omega * scale[None, :]


def generate(spec, n, seed):
    rng = np.random.default_rng(seed)
    factors = rng.multivariate_normal(np.zeros(len(spec.blocks)), spec.phi, size=n, method="cholesky")
    x = rng.normal(size=(n, spec.p))
    for h, block in enumerate(spec.blocks):
        index = list(block)
        signal = factors[:, [h]] if spec.loadings is None else factors[:, [h]] * spec.loadings[index]
        x[:, index] = signal + rng.normal(scale=np.sqrt(spec.residual[h]), size=(n, len(index)))
    return x


def relative_errors(estimate, truth):
    chol = np.linalg.cholesky(truth)
    whitened = np.linalg.solve(chol, np.linalg.solve(chol, estimate).T)
    whitened = (whitened + whitened.T) / 2
    return dict(whitened_frobenius=float(np.linalg.norm(whitened - np.eye(len(truth)))))
