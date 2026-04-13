"""Greedy algorithms for optimal experimental design.

Implements sequential design optimization using greedy selection with
numerically stable Sherman-Morrison updates for posterior covariance.
"""
from matplotlib import pyplot as plt
import numpy as np
from boed.design.criteria import DesignCriteria


def run_greedy_oed_LG(
    N,
    n_steps,
    u_true,
    model,
    prior_kernel,
    noise_model,
    candidates_x,
    candidates_t,
    n_budget,
    criterion_type="A",
    L_qoi=None,
    verbose: bool = True,
):
    """
    Greedy OED (linear / linearized) updating the posterior covariance
    via Sherman-Morrison.

    Notes
    -----
    - `u_true` is only used when the nonlinear model requires a linearization
      (e.g., Burgers with `get_forward_operator(t, u0_ref=u_true)`).
    - `criterion_type="EIG_LINEAR_OBS"` is implemented via KL-EIG on `Sigma_post`
      (ranking equivalent to D-opt in the linear-Gaussian case).
    """

    def _resolve_prior_covariance(N_local: int, prior_obj):
        if isinstance(prior_obj, np.ndarray):
            Sigma0 = np.asarray(prior_obj, dtype=float)
        elif hasattr(prior_obj, "Sigma"):
            Sigma0 = np.asarray(prior_obj.Sigma, dtype=float)
        elif callable(prior_obj):
            x = np.linspace(0.0, 1.0, N_local)
            Sigma0 = np.asarray(prior_obj(x, x), dtype=float)
            Sigma0 = Sigma0 + 1e-10 * np.eye(N_local)
        else:
            raise TypeError(
                "prior_kernel must be a covariance matrix, an object with '.Sigma', "
                "or a callable kernel k(x, x')."
            )

        if Sigma0.shape != (N_local, N_local):
            raise ValueError(
                f"Prior covariance shape mismatch: expected ({N_local}, {N_local}), got {Sigma0.shape}"
            )
        return Sigma0

    N = int(N)
    if hasattr(model, "N") and int(model.N) != N:
        raise ValueError(f"N ({N}) must match model.N ({model.N})")

    candidates_x = np.asarray(candidates_x, dtype=int)
    candidates_t = np.asarray(candidates_t, dtype=int)

    if candidates_x.size == 0 or candidates_t.size == 0:
        raise ValueError("candidates_x and candidates_t must be non-empty")
    if n_budget < 0:
        raise ValueError("n_budget must be >= 0")
    if int(np.max(candidates_t)) > int(n_steps):
        raise ValueError(
            f"Max candidate time ({int(np.max(candidates_t))}) exceeds n_steps ({n_steps})"
        )

    Sigma_prior = _resolve_prior_covariance(N, prior_kernel)
    current_Sigma = Sigma_prior.copy()
    selected_design = []
    history = []
    eig_linear_obs_cum = 0.0

    criterion = str(criterion_type).upper()

    # Noise variance per measurement (scalar or location-dependent vector)
    # Observation noise covariance (in physical space)
    Sigma_noise = np.asarray(noise_model.get_covariance(N), dtype=float)
    if Sigma_noise.shape != (N, N):
        raise ValueError(f"Noise covariance shape mismatch: got {Sigma_noise.shape}, expected ({N}, {N})")


    # Precompute forward operators G_t (shape (N, N)) for each candidate time
    unique_t = sorted(set(int(t) for t in candidates_t))
    forward_ops = {}

    if hasattr(model, "get_forward_operator"):
        for t in unique_t:
            try:
                Gt = model.get_forward_operator(t)
            except (TypeError, ValueError):
                if u_true is None:
                    raise ValueError(
                        "Model requires a reference state for linearization. "
                        "Pass u_true (or u0_ref) to run_greedy_oed_LG."
                    )
                Gt = model.get_forward_operator(t, u0_ref=u_true)

            Gt = np.asarray(Gt, dtype=float)
            if Gt.shape != (N, N):
                raise ValueError(f"Forward operator at t={t} has shape {Gt.shape}, expected ({N}, {N})")
            forward_ops[t] = Gt

    elif hasattr(model, "get_transition_matrix"):
        M = np.asarray(model.get_transition_matrix(), dtype=float)
        if M.shape != (N, N):
            raise ValueError(f"Transition matrix has shape {M.shape}, expected ({N}, {N})")
        for t in unique_t:
            forward_ops[t] = np.linalg.matrix_power(M, t)
    else:
        raise AttributeError(
            "Model must implement get_forward_operator(t[, u0_ref]) or get_transition_matrix()."
        )

    if verbose:
        print(f"--- Greedy OED LG optimization (criterion {criterion_type}) ---")

    for k in range(n_budget):
        best_score = np.inf
        best_cand = None
        best_Sigma_step = None

        for ti in candidates_t:
            ti = int(ti)
            Gt = forward_ops[ti]

            for xi in candidates_x:
                xi = int(xi)
                if (xi, ti) in selected_design:
                    continue

                # Ligne de mesure: y(x_i, t_i) = g @ theta + eps
                g = Gt[xi, :].reshape(1, -1)

                local_sigma2 = float(Sigma_noise[xi, xi])


                # Sherman-Morrison update
                S = (g @ current_Sigma @ g.T).item() + local_sigma2
                diff = (current_Sigma @ g.T) @ (g @ current_Sigma) / S
                Sigma_temp = current_Sigma - diff

                # Score
                if criterion == "A":
                    score = DesignCriteria.A_opt(Sigma_temp)
                elif criterion == "D":
                    score = DesignCriteria.D_opt(Sigma_temp)
                elif criterion == "C":
                    if L_qoi is None:
                        raise ValueError("L_qoi is required for criterion C")
                    score = DesignCriteria.C_opt(Sigma_temp, L_qoi)
                elif criterion == "EIG":
                    # Greedy minimizes -> maximize EIG via the negative sign
                    score = -DesignCriteria.EIG(Sigma_temp, Sigma_prior)
                elif criterion == "EIG_LINEAR_OBS":
                    # Incremental information gain for one scalar observation
                    # y = g @ theta + eps, eps ~ N(0, sigma^2)
                    # EIG = 0.5 * log(1 + g Sigma g^T / sigma^2)
                    if local_sigma2 <= 0.0:
                        raise ValueError(
                            f"Noise variance at x={xi} must be > 0 for EIG_LINEAR_OBS"
                        )
                    signal_var = float((g @ current_Sigma @ g.T).item())
                    eig_gain = 0.5 * np.log1p(signal_var / local_sigma2)
                    # Greedy minimizes -> maximize EIG via the negative sign
                    score = -eig_gain
                else:
                    raise ValueError(f"Unknown criterion type: {criterion_type}")

                if score < best_score:
                    best_score, best_cand, best_Sigma_step = score, (xi, ti), Sigma_temp

        if best_cand is None:
            break

        current_Sigma = best_Sigma_step
        selected_design.append(best_cand)

        if criterion == "EIG":
            history_value = -best_score
        elif criterion == "EIG_LINEAR_OBS":
            eig_linear_obs_cum += -best_score
            history_value = eig_linear_obs_cum
        else:
            history_value = best_score
        history.append(history_value)

        if verbose:
            print(
                f"Step {k+1}/{n_budget}: x={best_cand[0]}, "
                f"t={best_cand[1]} | Score: {history_value:.4e}"
            )

    return selected_design, history, current_Sigma

def run_greedy_oed(*args, **kwargs):
    """Greedy sequential experimental design optimization.
    
    Selects experimental measurements sequentially by greedily choosing the
    candidate that optimizes the design criterion at each step. Uses
    Sherman-Morrison updates for numerical efficiency.
    
    Parameters
    ----------
    model : ForwardModelBase
        PDE forward model with methods:
        - get_transition_matrix(): Returns time-stepping matrix
        - N: Number of spatial grid points
    Sigma_prior : np.ndarray
        Prior parameter covariance matrix, shape (N, N)
    noise_model : NoiseModelBase
        Noise model with attribute 'sigma' (noise standard deviation)
    candidates_x : array-like
        Spatial indices of candidate sensor locations
    candidates_t : array-like
        Temporal indices of candidate measurement times
    n_budget : int
        Number of measurements to select (budget)
    criterion_type : {'A', 'D', 'C', 'EIG'}, default='A'
        Design optimality criterion:
        - 'A': A-optimality (minimize trace of posterior covariance)
        - 'D': D-optimality (minimize log-determinant of posterior covariance)
        - 'C': C-optimality (minimize variance of linear functional)
        - 'EIG': maximize expected information gain
    L_qoi : np.ndarray, optional
        Observation matrix for quantity of interest (QoI).
        Required if criterion_type='C', shape (m, N) where m is number of QoIs
    max_per_time : int, optional
        Maximum number of selected sensors per time index.
        If None, no per-time limit is applied.
        
    Returns
    -------
    selected_design : list of tuple
        List of (x_index, t_index) pairs selected in order
    history : list of float
        Design criterion value at each step
    current_Sigma : np.ndarray
        Final posterior covariance after all selections, shape (N, N)
        
    Notes
    -----
    **Algorithm:**
    1. Compute trajectory operator T for all candidate time steps
    2. At each iteration k=1...n_budget:
       a. For each unselected candidate (xi, ti):
          - Extract measurement vector g from trajectory operator
          - Use Sherman-Morrison formula to update posterior covariance
          - Evaluate design criterion on updated covariance
       b. Select candidate with best (smallest) criterion value
       c. Store criterion value in history
    
    **Sherman-Morrison Update:**
    For measurement g with noise variance σ²:
    
    .. math::
        Σ_new = Σ - (Σg^T g Σ) / (g^T Σ g + σ²)
    
    This is numerically stable and avoids explicit matrix inversion.
    
    **Computational Complexity:**
    - Per step: O(n_cand · N²) where n_cand = len(candidates_x) × len(candidates_t)
    - Total: O(n_budget · n_cand · N²)
    
    Examples
    --------
    >>> from boed.design.greedy import run_greedy_oed
    >>> from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
    >>> from boed.priors.gp_priors import GaussianProcessPrior
    >>> from boed.priors.kernels import Gaussian
    >>> from boed.core.noise import NoiseModel
    >>> import numpy as np
    >>> 
    >>> # Setup
    >>> N, dt, n_steps = 50, 0.01, 20
    >>> model = AdvectionDiffusion1D_CN(N, dt, diffusivity=0.01, velocity=0.5)
    >>> kernel = Gaussian(length_scale=0.2, sigma=1.0)
    >>> prior = GaussianProcessPrior(kernel, nx=N)
    >>> noise = NoiseModel(sigma_noise=0.01)
    >>> 
    >>> # Greedy design selection
    >>> design, history, Sigma = run_greedy_oed(
    ...     model=model,
    ...     Sigma_prior=prior.Sigma,
    ...     noise_model=noise,
    ...     candidates_x=np.arange(N),
    ...     candidates_t=np.arange(n_steps),
    ...     n_budget=10,
    ...     criterion_type="A"
    ... )
    >>> print(f"Selected {len(design)} measurements")
    >>> print(f"Final A-optimality: {history[-1]:.4e}")
    """
    kwargs = dict(kwargs)

    # Backward-compatible parsing:
    # - Canonical: run_greedy_oed(model, Sigma_prior, noise_model, ...)
    # - Legacy positional: run_greedy_oed(N, model, prior_kernel, noise_model, ...)
    # - Legacy keywords: run_greedy_oed(N=..., model=..., prior_kernel=..., ...)
    canonical_order = [
        "model",
        "Sigma_prior",
        "noise_model",
        "candidates_x",
        "candidates_t",
        "n_budget",
        "criterion_type",
        "L_qoi",
        "verbose",
        "max_per_time",
    ]
    legacy_order = [
        "N",
        "model",
        "prior_kernel",
        "noise_model",
        "candidates_x",
        "candidates_t",
        "n_budget",
        "criterion_type",
        "L_qoi",
        "verbose",
        "max_per_time",
    ]

    legacy_mode = ("N" in kwargs) or ("prior_kernel" in kwargs)
    if not legacy_mode and len(args) >= 1 and isinstance(args[0], (int, np.integer)):
        legacy_mode = True

    arg_order = legacy_order if legacy_mode else canonical_order
    if len(args) > len(arg_order):
        raise TypeError(
            f"run_greedy_oed() takes at most {len(arg_order)} positional arguments "
            f"({len(args)} given)"
        )

    values = {}
    for name, value in zip(arg_order, args):
        if name in kwargs:
            raise TypeError(f"run_greedy_oed() got multiple values for argument '{name}'")
        values[name] = value

    for name in arg_order[len(args) :]:
        if name in kwargs:
            values[name] = kwargs.pop(name)

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"run_greedy_oed() got unexpected keyword argument(s): {unknown}")

    model = values.get("model")
    Sigma_prior = values.get("Sigma_prior")
    if legacy_mode and Sigma_prior is None:
        Sigma_prior = values.get("prior_kernel")
    noise_model = values.get("noise_model")
    candidates_x = values.get("candidates_x")
    candidates_t = values.get("candidates_t")
    n_budget = values.get("n_budget")
    criterion_type = values.get("criterion_type", "A")
    L_qoi = values.get("L_qoi", None)
    verbose = values.get("verbose", True)
    max_per_time = values.get("max_per_time", None)
    N_legacy = values.get("N", None)

    missing = [
        name
        for name, value in [
            ("model", model),
            ("Sigma_prior", Sigma_prior),
            ("noise_model", noise_model),
            ("candidates_x", candidates_x),
            ("candidates_t", candidates_t),
            ("n_budget", n_budget),
        ]
        if value is None
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise TypeError(f"run_greedy_oed() missing required argument(s): {missing_str}")

    N = int(model.N)
    if N_legacy is not None and int(N_legacy) != N:
        raise ValueError(f"N ({int(N_legacy)}) must match model.N ({N})")

    current_Sigma = Sigma_prior.copy()
    selected_design = []
    selected_set = set()
    history = []
    criterion = str(criterion_type).upper()
    eig_cum = 0.0
    time_counts = {}
    if max_per_time is not None:
        max_per_time = int(max_per_time)
        if max_per_time <= 0:
            raise ValueError("max_per_time must be >= 1 when provided.")

    # Precompute trajectory operator
    M = model.get_transition_matrix()
    max_t = int(np.max(candidates_t))
    Trajectory_Op = np.vstack([np.linalg.matrix_power(M, t) for t in range(max_t + 1)])
    
    sigma2 = getattr(noise_model, 'sigma', 0.01)**2
    if criterion == "EIG" and sigma2 <= 0.0:
        raise ValueError("Noise variance must be > 0 for criterion 'EIG'.")

    if verbose:
        print(f"--- Greedy OED optimization (criterion {criterion}) ---")

    for k in range(n_budget):
        best_score = np.inf
        best_cand = None
        best_Sigma_step = None
        best_eig_gain = None
        
        for ti in candidates_t:
            for xi in candidates_x:
                if (xi, ti) in selected_set:
                    continue
                if max_per_time is not None and time_counts.get(int(ti), 0) >= max_per_time:
                    continue
                
                # Sherman-Morrison update for Sigma
                g = Trajectory_Op[ti * N + xi, :].reshape(1, -1)
                signal_var = (g @ current_Sigma @ g.T).item()
                S = signal_var + sigma2
                diff = (current_Sigma @ g.T) @ (g @ current_Sigma) / S
                Sigma_temp = current_Sigma - diff
                
                # Score computation
                if criterion == "A":
                    score = DesignCriteria.A_opt(Sigma_temp)
                elif criterion == "D":
                    score = DesignCriteria.D_opt(Sigma_temp)
                elif criterion == "C":
                    if L_qoi is None: 
                        raise ValueError("L_qoi is required for criterion C")
                    score = DesignCriteria.C_opt(Sigma_temp, L_qoi)
                elif criterion == "EIG":
                    # Exact rank-1 EIG increment for scalar linear observation:
                    # Delta EIG = 0.5 * log(1 + g Sigma g^T / sigma^2)
                    eig_gain = 0.5 * np.log1p(signal_var / sigma2)
                    # Greedy minimizes score -> maximize cumulative EIG.
                    score = -(eig_cum + eig_gain)
                else:
                    raise ValueError(f"Unknown criterion type: {criterion_type}")
                
                if score < best_score:
                    best_score, best_cand, best_Sigma_step = score, (xi, ti), Sigma_temp
                    if criterion == "EIG":
                        best_eig_gain = eig_gain

        if best_cand is None: 
            break

        current_Sigma = best_Sigma_step
        selected_design.append(best_cand)
        selected_set.add(best_cand)
        t_sel = int(best_cand[1])
        time_counts[t_sel] = time_counts.get(t_sel, 0) + 1
        if criterion == "EIG":
            # Keep history as cumulative EIG in nats.
            eig_cum += float(best_eig_gain if best_eig_gain is not None else 0.0)
            history.append(eig_cum)
        else:
            history.append(best_score)
        
        if verbose:
            print(
                f"Step {k+1}/{n_budget}: x={best_cand[0]}, "
                f"t={best_cand[1]} | Score: {history[-1]:.4e}"
            )

    return selected_design, history, current_Sigma


def run_sequential_step_oed(
    model,
    Sigma_prior,
    noise_model,
    candidates_x,
    times_to_observe,  # List of time steps, e.g. [0, 5, 10...]
    budget_per_step,   # Number of sensors to place at EACH time
    criterion_type="A",
    L_qoi=None
):
    N = model.N
    current_Sigma = Sigma_prior.copy()
    selected_design = []
    history = []
    
    # Precomputation
    M = model.get_transition_matrix()
    sigma2 = getattr(noise_model, 'sigma', 0.01)**2

    for ti in times_to_observe:
        # Transfer operator for the specific time ti
        T_ti = np.linalg.matrix_power(M, ti)
        
        # Place budget_per_step sensors at this specific time
        for k in range(budget_per_step):
            best_score = np.inf
            best_xi = None
            best_Sigma_temp = None
            
            for xi in candidates_x:
                if (xi, ti) in selected_design: continue
                
                # Local measurement at time ti
                g = T_ti[xi, :].reshape(1, -1)
                
                # Sherman-Morrison update
                S = (g @ current_Sigma @ g.T).item() + sigma2
                diff = (current_Sigma @ g.T) @ (g @ current_Sigma) / S
                Sigma_temp = current_Sigma - diff
                
                # Score computation according to the criterion
                if criterion_type == "C":
                    score = DesignCriteria.C_opt(Sigma_temp, L_qoi)
                else: # A-opt by default
                    score = np.trace(Sigma_temp)
                
                if score < best_score:
                    best_score, best_xi, best_Sigma_temp = score, xi, Sigma_temp
            
            if best_xi is not None:
                selected_design.append((best_xi, ti))
                current_Sigma = best_Sigma_temp
                history.append(best_score)
                
    return selected_design, history, current_Sigma


def run_sboed_trajectories(model, Sigma_prior, noise_model, candidates_x,
                            times, budget_per_step, L_qoi=None):
    N = model.N
    current_Sigma = Sigma_prior.copy()
    trajectory_history = [] # Store (t, x_selected)
    
    M = model.get_transition_matrix()
    sigma2 = getattr(noise_model, 'sigma', 0.01)**2

    for t_idx in times:
        # 1. Compute the operator for the current time t
        T_t = np.linalg.matrix_power(M, t_idx)
        
        step_sensors = []
        for b in range(budget_per_step):
            best_score = np.inf
            best_x = None
            best_Sigma_update = None
            
            for xi in candidates_x:
                if xi in step_sensors: continue
                
                # Measurement vector g depends on the physics at time t
                g = T_t[xi, :].reshape(1, -1)
                
                # Sherman-Morrison update to simulate data acquisition
                S = (g @ current_Sigma @ g.T).item() + sigma2
                diff = (current_Sigma @ g.T) @ (g @ current_Sigma) / S
                Sigma_temp = current_Sigma - diff
                
                # C criterion (goal-oriented) or A criterion (global)
                if L_qoi is not None:
                    score = np.trace(L_qoi @ Sigma_temp @ L_qoi.T) # C-opt variance
                else:
                    score = np.trace(Sigma_temp) # A-opt
                
                if score < best_score:
                    best_score, best_x, best_Sigma_update = score, xi, Sigma_temp
            
            if best_x is not None:
                step_sensors.append(best_x)
                current_Sigma = best_Sigma_update
                trajectory_history.append((t_idx, best_x))
                
    return np.array(trajectory_history), current_Sigma
