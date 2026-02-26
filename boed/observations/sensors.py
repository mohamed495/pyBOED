"""Observation operators for spatial and spatio-temporal measurements.

This module defines *observation operators* that map a state (or stacked
trajectory state) to measurements. This is distinct from
``boed.inference.ObservationMap``, which operates in observation space for
selection/compression of an already-defined observation vector.
"""
import numpy as np
from boed.core.base import validate_design_indices


class SpaceTimeSensors:
    """Manages spatio-temporal sensor placements and observation operators.

    Creates row-selection operators that extract values at specific space-time
    points from a flattened trajectory/state vector. Validates uniqueness of
    measurement locations and provides conversion helpers for design and
    inference workflows.
    
    Parameters
    ----------
    x_idx : array-like
        Spatial indices of sensors, shape (n_sensors,)
    t_idx : array-like
        Temporal indices of sensors, shape (n_sensors,)
    nx : int
        Number of spatial grid points
        
    Attributes
    ----------
    x_idx : array-like
        Spatial indices
    t_idx : array-like
        Temporal indices
    nx : int
        Number of spatial grid points
        
    Raises
    ------
    ValueError
        If the same space-time point is measured twice
        
    Examples
    --------
    Place 10 sensors at random space-time locations:
    
    >>> from boed.observations.sensors import SpaceTimeSensors
    >>> import numpy as np
    >>> 
    >>> # Grid with 50 spatial points, 20 time steps
    >>> nx, nt = 50, 20
    >>> 
    >>> # Random sensor placement
    >>> x_idx = np.random.randint(0, nx, 10)
    >>> t_idx = np.random.randint(0, nt, 10)
    >>> sensors = SpaceTimeSensors(x_idx, t_idx, nx)
    >>> 
    >>> # Generate observation operator (canonical name)
    >>> H_obs = sensors.observation_operator(nt)
    >>> print(H_obs.shape)  # (10, 50*20)
    
    >>> # Extract observations y = H_obs @ u_flattened
    """
    
    def __init__(self, x_idx, t_idx, nx):
        """Initialize sensor configuration.
        
        Parameters
        ----------
        x_idx : array-like
            Spatial indices of sensors
        t_idx : array-like
            Temporal indices of sensors
        nx : int
            Number of spatial grid points
            
        Raises
        ------
        ValueError
            If duplicate (x, t) pairs are provided
        """
        self.nx = int(nx)
        if self.nx <= 0:
            raise ValueError("nx must be positive")

        x_arr = np.asarray(x_idx, dtype=int).ravel()
        t_arr = np.asarray(t_idx, dtype=int).ravel()
        if x_arr.size == 0 or t_arr.size == 0:
            raise ValueError("x_idx and t_idx must be non-empty")
        if x_arr.size != t_arr.size:
            raise ValueError("x_idx and t_idx must have the same length")
        validate_design_indices(x_arr, self.nx, allow_duplicates=True)
        if np.any(t_arr < 0):
            raise ValueError("Temporal indices must be non-negative")

        # Validation: ensure unique space-time sensor locations
        design_points = list(zip(x_arr.tolist(), t_arr.tolist()))
        if len(set(design_points)) < len(design_points):
            raise ValueError("Duplicate space-time indices not allowed")
            
        self.x_idx = x_arr
        self.t_idx = t_arr

    @classmethod
    def from_design(cls, design, nx: int) -> "SpaceTimeSensors":
        """Build a sensor object from a design list of ``(x_idx, t_idx)`` pairs.

        This is the native output format of ``boed.design.selection`` utilities.
        """
        pairs = list(design)
        if len(pairs) == 0:
            raise ValueError("design must contain at least one (x_idx, t_idx) pair")
        try:
            x_idx, t_idx = zip(*pairs)
        except ValueError as exc:
            raise ValueError("design must be an iterable of (x_idx, t_idx) pairs") from exc
        return cls(x_idx=x_idx, t_idx=t_idx, nx=nx)

    def as_design(self) -> list[tuple[int, int]]:
        """Return the sensor design as a list of ``(x_idx, t_idx)`` pairs."""
        return [(int(xi), int(ti)) for xi, ti in zip(self.x_idx, self.t_idx)]

    def flattened_indices(self, nt: int) -> np.ndarray:
        """Return flattened trajectory indices for the selected space-time points.

        Indices follow the same ordering assumed by :meth:`observation_operator`:
        ``u_flat = [u(t=0), u(t=1), ...]`` with each block of length ``nx``.
        """
        nt = int(nt)
        if nt <= 0:
            raise ValueError("nt must be positive")
        if np.any(self.t_idx >= nt):
            raise ValueError(f"Temporal indices must be in [0, {nt-1}] for nt={nt}.")
        return (self.t_idx * self.nx + self.x_idx).astype(int, copy=False)

    def as_observation_map(self, nt: int):
        """Return an ``ObservationMap`` for the flattened trajectory vector.

        This is useful when a trajectory has already been vectorized and you
        want to reuse the inference-side ``ObservationMap`` abstraction.
        """
        from boed.inference.observation_map import ObservationMap

        n_obs = self.nx * int(nt)
        return ObservationMap.from_indices(self.flattened_indices(nt), n_obs=n_obs)

    def observation_operator(self, nt: int) -> np.ndarray:
        """Create an observation operator for extracting measurements.
        
        Generates a binary row-selection operator ``H_obs`` that extracts values at specified
        space-time locations from the flattened state vector.
        
        The state vector is assumed to be ordered as:
        u = [u_0(t=0), u_1(t=0), ..., u_0(t=1), u_1(t=1), ...]
        
        Parameters
        ----------
        nt : int
            Number of time steps
            
        Returns
        -------
        H_obs : np.ndarray
            Observation operator, shape ``(n_sensors, nx*nt)``.
            Each row extracts one sensor measurement.
            
        Notes
        -----
        Observation equation: ``y = H_obs @ u_flat``
        where u_flat is the flattened state at all times and locations.
        """
        flat_idx = self.flattened_indices(nt)
        n_sensors = len(flat_idx)
        H_obs = np.zeros((n_sensors, self.nx * int(nt)))
        for i, row_idx in enumerate(flat_idx):
            # Global flattened index: time step * nx + spatial index
            H_obs[i, int(row_idx)] = 1.0
        return H_obs

    def observation_matrix(self, nt: int) -> np.ndarray:
        """Legacy alias for :meth:`observation_operator`."""
        return self.observation_operator(nt)
