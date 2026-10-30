"""Constraint-preserving, explicitly qualified bagged covariance calibration."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JointCovariance:
    labels: tuple
    free_labels: tuple
    raw: np.ndarray
    adjusted: np.ndarray
    stabilized: np.ndarray
    free_raw: np.ndarray
    free_adjusted: np.ndarray
    free_stabilized: np.ndarray
    affine_map: np.ndarray
    block_ids: np.ndarray
    random_state: int
    draws: int
    qualification: str


def stabilize_fixed_diagonal(covariance, floor=1e-10):
    """PSD projection in free coordinates, leaving deterministic zeros zero."""
    c = (covariance + covariance.T) / 2
    diagonal = np.maximum(np.diag(c), 0.0)
    active = diagonal > 0
    result = np.zeros_like(c)
    if not np.any(active):
        return result
    scale = np.sqrt(diagonal[active])
    correlation = c[np.ix_(active, active)] / np.outer(scale, scale)
    values, vectors = np.linalg.eigh((correlation + correlation.T) / 2)
    correlation = (vectors * np.maximum(values, floor)) @ vectors.T
    normalizer = np.sqrt(np.diag(correlation))
    correlation /= np.outer(normalizer, normalizer)
    result[np.ix_(active, active)] = correlation * np.outer(scale, scale)
    return (result + result.T) / 2


def calibrate_free_covariance(covariance, block_ids, ratio):
    """Moment calibration: c/2 within retained q-blocks, c between q-blocks.

    Scalar independent coordinates recover the historical diagonal/off-diagonal
    rule. Structured loading blocks and the full q(Phi) block are retained.
    This is an implementation-level block extension, not a proved calibration
    theorem for discrete sparse loadings; the raw mixture remains authoritative.
    """
    blocks = np.asarray(block_ids)
    same = blocks[:, None] == blocks[None, :]
    return np.asarray(covariance) * np.where(same, ratio / 2, ratio)


def coordinate_layout(posterior, parameters):
    """Affine full-parameter coordinates from nonredundant loading coordinates."""
    if not getattr(posterior, "enforce_loading_row_sums", True):
        raise NotImplementedError(
            "joint-moment coordinates are defined for loading_normalization='row_sum' only"
        )
    p, k = posterior.loading_mean.shape
    primary = np.argmax(posterior.loading_mask, axis=1)
    columns, labels, free_labels, blocks = [], [], [], []
    offset = 0
    group = 0
    selections = {}
    for parameter in parameters:
        if posterior.member == "1f" and parameter == "Lambda":
            selections[parameter] = np.array([], dtype=int)
            for row in range(p):
                for factor in range(k):
                    labels.append(f"Lambda[{row},{factor}]")
                    columns.append({})
            continue
        if posterior.member == "1f" and parameter == "psi":
            first = np.array([np.flatnonzero(primary == h)[0] for h in range(k)])
            selections[parameter] = first
            for h in range(k):
                free_labels.append(f"v[{h}]")
                blocks.append(group+h)
            for row in range(p):
                labels.append(f"psi[{row}]")
                columns.append({offset+int(primary[row]): 1.0})
            offset += k
            group += k
            continue
        if parameter == "Lambda":
            cross = np.flatnonzero(~posterior.loading_mask.astype(bool).reshape(-1))
            selections[parameter] = cross
            lookup = {int(flat): offset + j for j, flat in enumerate(cross)}
            for flat in cross:
                row, factor = divmod(int(flat), k)
                free_labels.append(f"Lambda[{row},{factor}]")
                if posterior.member == "mf":
                    blocks.append(group)
                    group += 1
                elif posterior.member == "2f":
                    blocks.append(group + row)
                else:
                    blocks.append(group + int(primary[row]))
            if posterior.member == "2f":
                group += p
            if posterior.member == "2f+s":
                group += k
            for row in range(p):
                for factor in range(k):
                    labels.append(f"Lambda[{row},{factor}]")
                    columns.append(
                        {lookup[row * k + factor]: 1.0}
                        if factor != primary[row]
                        else {
                            lookup[row * k + t]: -1.0
                            for t in range(k)
                            if t != primary[row]
                        }
                    )
            offset += len(cross)
        else:
            selected = (
                np.ravel_multi_index(np.triu_indices(k), (k, k))
                if parameter == "Phi"
                else np.arange(p)
            )
            selections[parameter] = selected
            for j, flat in enumerate(selected):
                label = (
                    f"Phi[{int(flat)//k},{int(flat)%k}]"
                    if parameter == "Phi"
                    else f"psi[{int(flat)}]"
                )
                labels.append(label)
                free_labels.append(label)
                columns.append({offset + j: 1.0})
                blocks.append(group)
                if parameter == "psi":
                    group += 1
            if parameter == "Phi":
                group += 1
            offset += len(selected)
    return labels, free_labels, blocks, columns, selections, offset
