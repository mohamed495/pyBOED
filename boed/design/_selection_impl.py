"""Sensor selection utilities (QR pivoting and max-volume heuristics).

These functions return designs as lists of ``(x_idx, t_idx)`` pairs. To build a
measurement operator from a selected design, use
``boed.observations.SpaceTimeSensors.from_design(...)``.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import numpy as np

from boed.design.criteria import DesignCriteria
from boed.design._greedy_impl import run_greedy_oed


def _as_int_vector(values: Iterable[int], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=int).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one index")
    return arr


def _trajectory_operator(model, max_t: int) -> np.ndarray:
    """Stacked trajectory operator: [M^0; M^1; ...; M^max_t]."""
    M = model.get_transition_matrix()
    return np.vstack([np.linalg.matrix_power(M, t) for t in range(max_t + 1)])


def _candidate_matrix(
    model,
    candidates_x: Iterable[int],
    candidates_t: Iterable[int],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Build candidate row matrix and (x, t) mapping."""
    N = model.N
    cand_x = _as_int_vector(candidates_x, "candidates_x")
    cand_t = _as_int_vector(candidates_t, "candidates_t")

    if np.any(cand_x < 0) or np.any(cand_x >= N):
        raise ValueError("candidates_x must be within [0, N-1]")
    if np.any(cand_t < 0):
        raise ValueError("candidates_t must be non-negative")

    max_t = int(np.max(cand_t))
    trajectory_op = _trajectory_operator(model, max_t=max_t)

    pairs: List[Tuple[int, int]] = []
    rows: List[int] = []
    seen = set()
    for ti in cand_t:
        for xi in cand_x:
            pair = (int(xi), int(ti))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            rows.append(int(ti) * N + int(xi))

    A = trajectory_op[rows, :]
    return A, pairs


def _pivoted_row_order(A: np.ndarray, n_select: Optional[int] = None) -> np.ndarray:
    """
    Return a greedy row-pivot order using modified Gram-Schmidt.

    This avoids relying on SciPy's pivoted QR in environments where LAPACK
    pivoting kernels may be unstable.
    """
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    m, _ = A.shape
    if n_select is None:
        n_select = m
    n_select = int(max(0, min(n_select, m)))

    work = np.asarray(A, dtype=float)
    norms2 = np.einsum("ij,ij->i", work, work).copy()
    selected: List[int] = []
    Q: List[np.ndarray] = []

    for _ in range(n_select):
        idx = int(np.argmax(norms2))
        if not np.isfinite(norms2[idx]) or norms2[idx] <= 1e-14:
            break

        v = work[idx].copy()
        for q in Q:
            v -= (q @ v) * q
        nrm = np.linalg.norm(v)
        if nrm <= 1e-14:
            norms2[idx] = -np.inf
            continue

        q = v / nrm
        Q.append(q)
        selected.append(idx)
        norms2[idx] = -np.inf

        proj = work @ q
        norms2 = np.maximum(norms2 - proj * proj, 0.0)
        for s in selected:
            norms2[s] = -np.inf

    if len(selected) < m:
        remaining = [i for i in range(m) if i not in set(selected)]
        remaining.sort(key=lambda i: norms2[i], reverse=True)
        selected.extend(remaining)

    return np.asarray(selected, dtype=int)


def select_sensors_qr_pivot(
    model,
    candidates_x: Iterable[int],
    candidates_t: Iterable[int],
    n_budget: int,
) -> List[Tuple[int, int]]:
    """Select sensors via QR pivoting on candidate rows.

    Returns
    -------
    list[tuple[int, int]]
        Selected ``(x_idx, t_idx)`` pairs. Pass directly to
        ``SpaceTimeSensors.from_design(selected, nx=model.N)`` to build an
        observation operator.
    """
    if n_budget <= 0:
        raise ValueError("n_budget must be positive")

    A, pairs = _candidate_matrix(model, candidates_x, candidates_t)
    if n_budget > A.shape[0]:
        raise ValueError("n_budget exceeds number of candidate points")

    piv = _pivoted_row_order(A, n_select=n_budget)
    selected = [pairs[int(i)] for i in piv[:n_budget]]
    return selected


def _maxvol_rows(
    U: np.ndarray,
    max_iters: int = 100,
    tol: float = 1.05,
) -> List[int]:
    """Heuristic max-volume row selection for a tall matrix U."""
    m, k = U.shape
    if k > m:
        raise ValueError("U must have at least as many rows as columns")

    piv = _pivoted_row_order(U, n_select=k)
    idx = [int(i) for i in piv[:k]]

    for _ in range(max_iters):
        A = U[idx, :]
        try:
            C = np.linalg.solve(A.T, U.T).T
        except np.linalg.LinAlgError:
            piv = _pivoted_row_order(U, n_select=k)
            idx = [int(i) for i in piv[:k]]
            A = U[idx, :]
            C = np.linalg.solve(A.T, U.T).T

        abs_c = np.abs(C)
        i, j = np.unravel_index(np.argmax(abs_c), abs_c.shape)
        if abs_c[i, j] <= tol:
            break
        idx[j] = int(i)

    return idx


def select_sensors_maxvol(
    model,
    candidates_x: Iterable[int],
    candidates_t: Iterable[int],
    n_budget: int,
    basis: str = "svd",
    max_iters: int = 100,
    tol: float = 1.05,
) -> List[Tuple[int, int]]:
    """Select sensors using a max-volume heuristic on a low-rank basis.

    Returns
    -------
    list[tuple[int, int]]
        Selected ``(x_idx, t_idx)`` pairs. Pass directly to
        ``SpaceTimeSensors.from_design(selected, nx=model.N)`` to build an
        observation operator.
    """
    if n_budget <= 0:
        raise ValueError("n_budget must be positive")

    A, pairs = _candidate_matrix(model, candidates_x, candidates_t)
    m, n = A.shape
    if n_budget > min(m, n):
        raise ValueError("n_budget exceeds feasible max-volume size")

    basis = basis.lower()
    if basis == "svd":
        U, _, _ = np.linalg.svd(A, full_matrices=False)
        U_k = U[:, :n_budget]
    elif basis == "qr":
        Q, _ = np.linalg.qr(A, mode="reduced")
        U_k = Q[:, :n_budget]
    else:
        raise ValueError("basis must be 'svd' or 'qr'")

    idx = _maxvol_rows(U_k, max_iters=max_iters, tol=tol)
    selected = [pairs[int(i)] for i in idx]
    return selected


def evaluate_design(
    model,
    Sigma_prior: np.ndarray,
    noise_model,
    design: Sequence[Tuple[int, int]],
    criterion_type: Optional[str] = None,
    L_qoi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Apply a fixed design and return posterior covariance and score history."""
    Sigma = Sigma_prior.copy()
    history: List[float] = []

    if len(design) == 0:
        return Sigma, history

    max_t = int(max(t for _, t in design))
    trajectory_op = _trajectory_operator(model, max_t=max_t)
    N = model.N
    sigma2 = getattr(noise_model, "sigma", 0.01) ** 2

    for xi, ti in design:
        g = trajectory_op[int(ti) * N + int(xi), :].reshape(1, -1)
        S = (g @ Sigma @ g.T).item() + sigma2
        diff = (Sigma @ g.T) @ (g @ Sigma) / S
        Sigma = Sigma - diff

        if criterion_type is not None:
            score = score_design(Sigma, Sigma_prior, criterion_type, L_qoi=L_qoi)
            history.append(float(score))

    return Sigma, history


def score_design(
    Sigma_post: np.ndarray,
    Sigma_prior: np.ndarray,
    criterion_type: str,
    L_qoi: Optional[np.ndarray] = None,
) -> float:
    """Score a design using the specified criterion."""
    criterion = criterion_type.upper()
    if criterion == "A":
        return DesignCriteria.A_opt(Sigma_post)
    if criterion == "D":
        return DesignCriteria.D_opt(Sigma_post)
    if criterion == "C":
        if L_qoi is None:
            raise ValueError("L_qoi required for C-optimality scoring")
        return DesignCriteria.C_opt(Sigma_post, L_qoi)
    if criterion == "EIG":
        return DesignCriteria.EIG(Sigma_post, Sigma_prior)
    if criterion == "EIG_LINEAR_OBS":
        # For linear-Gaussian sensor selection, this is equivalent to D-opt
        # up to an additive constant from the fixed prior covariance.
        return DesignCriteria.EIG(Sigma_post, Sigma_prior)
    raise ValueError(f"Unknown criterion type: {criterion_type}")


def compare_to_greedy(
    model,
    Sigma_prior: np.ndarray,
    noise_model,
    candidates_x: Iterable[int],
    candidates_t: Iterable[int],
    n_budget: int,
    criterion_type: str = "D",
    L_qoi: Optional[np.ndarray] = None,
    maxvol_basis: str = "svd",
) -> Dict[str, Dict[str, object]]:
    """Compare QR-pivot and max-volume selections against greedy OED."""
    if criterion_type.upper() == "EIG":
        raise ValueError("Greedy comparison currently supports only A/D/C criteria")
    greedy_design, greedy_history, greedy_Sigma = run_greedy_oed(
        model=model,
        Sigma_prior=Sigma_prior,
        noise_model=noise_model,
        candidates_x=candidates_x,
        candidates_t=candidates_t,
        n_budget=n_budget,
        criterion_type=criterion_type,
        L_qoi=L_qoi,
    )
    greedy_score = score_design(greedy_Sigma, Sigma_prior, criterion_type, L_qoi=L_qoi)

    qr_design = select_sensors_qr_pivot(
        model=model,
        candidates_x=candidates_x,
        candidates_t=candidates_t,
        n_budget=n_budget,
    )
    qr_Sigma, qr_history = evaluate_design(
        model=model,
        Sigma_prior=Sigma_prior,
        noise_model=noise_model,
        design=qr_design,
        criterion_type=criterion_type,
        L_qoi=L_qoi,
    )
    qr_score = score_design(qr_Sigma, Sigma_prior, criterion_type, L_qoi=L_qoi)

    maxvol_design = select_sensors_maxvol(
        model=model,
        candidates_x=candidates_x,
        candidates_t=candidates_t,
        n_budget=n_budget,
        basis=maxvol_basis,
    )
    maxvol_Sigma, maxvol_history = evaluate_design(
        model=model,
        Sigma_prior=Sigma_prior,
        noise_model=noise_model,
        design=maxvol_design,
        criterion_type=criterion_type,
        L_qoi=L_qoi,
    )
    maxvol_score = score_design(maxvol_Sigma, Sigma_prior, criterion_type, L_qoi=L_qoi)

    return {
        "greedy": {
            "design": greedy_design,
            "history": greedy_history,
            "score": float(greedy_score),
            "Sigma_post": greedy_Sigma,
        },
        "qr": {
            "design": qr_design,
            "history": qr_history,
            "score": float(qr_score),
            "Sigma_post": qr_Sigma,
        },
        "maxvol": {
            "design": maxvol_design,
            "history": maxvol_history,
            "score": float(maxvol_score),
            "Sigma_post": maxvol_Sigma,
        },
    }
