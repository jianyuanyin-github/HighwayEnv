from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import numpy as np
from torch import Tensor


@dataclass
class Trial:
    """Data class for storing trial information."""

    id: int
    params_nat: Tensor  # natural (unnormalized) parameters
    params_nor: Tensor  # normalized parameters
    objectives: Tensor  # objective values
    feasible: bool  # feasibility indicator

    def __str__(self) -> str:
        """String representation of trial."""
        param_str = ", ".join([f"{p:.3f}" for p in self.params_nat])
        obj_str = ", ".join([f"{o:.3f}" for o in self.objectives])
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        return f"Trial {self.id}: [{param_str}] -> [{obj_str}] ({status})"


@dataclass
class Dataset:
    """Data class for storing dataset information."""

    X: Tensor  # input parameters (normalized)
    Y: Tensor  # output objectives

    def __post_init__(self):
        """Validate dataset dimensions."""
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have same number of samples")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.X.shape[0]

    def add_trial(self, trial: Trial):
        """Add trial to dataset."""
        if trial.feasible:
            self.X = torch.cat([self.X, trial.params_nor.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, trial.objectives.unsqueeze(0)], dim=0)

    def get_pareto_front(self) -> Tuple[Tensor, Tensor]:
        """Get Pareto optimal points from dataset."""
        if len(self) == 0:
            return torch.empty((0, self.X.shape[1])), torch.empty((0, self.Y.shape[1]))

        # Find non-dominated points
        pareto_mask = self._get_pareto_mask()
        pareto_X = self.X[pareto_mask]
        pareto_Y = self.Y[pareto_mask]

        return pareto_X, pareto_Y

    def _get_pareto_mask(self) -> Tensor:
        """Get boolean mask for Pareto optimal points."""
        if len(self) == 0:
            return torch.empty(0, dtype=torch.bool)

        # For minimization problem, find non-dominated points
        n_points = len(self)
        pareto_mask = torch.ones(n_points, dtype=torch.bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if torch.all(self.Y[j] <= self.Y[i]) and torch.any(
                        self.Y[j] < self.Y[i]
                    ):
                        pareto_mask[i] = False
                        break

        return pareto_mask


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""

    n_initial: int
    n_bayesian_optimization: int
    batch_size: int
    n_processes: int
    max_lat_dev: float
    max_a_comb: float
    reference_points: list
    epsilon: float
    n_sobol_samples: int
    n_restarts: int
    n_raw_samples: int
    max_iter: int

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(
            n_initial=config_dict["n_initial"],
            n_bayesian_optimization=config_dict["n_bayesian_optimization"],
            batch_size=config_dict["batch_size"],
            n_processes=config_dict["n_processes"],
            max_lat_dev=config_dict["max_lat_dev"],
            max_a_comb=config_dict["max_a_comb"],
            reference_points=[
                config_dict["reference_point_0"],
                config_dict["reference_point_1"],
            ],
            epsilon=config_dict["epsilon"],
            n_sobol_samples=config_dict["n_sobol_samples"],
            n_restarts=config_dict["n_restarts"],
            n_raw_samples=config_dict["n_raw_samples"],
            max_iter=config_dict["max_iter"],
        )


@dataclass
class OptimizationState:
    """State of the optimization process."""
    
    train_X: Tensor = None
    train_Y: Tensor = None
    train_C: Tensor = None
    
    def __post_init__(self):
        """Initialize empty tensors if None."""
        if self.train_X is None:
            self.train_X = torch.empty((0, 5))  # 5 parameters: Qc, Ql, Q_theta, R_d, R_delta
        if self.train_Y is None:
            self.train_Y = torch.empty((0, 2))  # 2 objectives: lap_time, tracking_error
        if self.train_C is None:
            self.train_C = torch.empty((0, 1))  # No constraint used (always feasible)


@dataclass
class BOResults:
    """Results from Bayesian optimization."""
    
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize empty arrays if None."""
        if self.X is None:
            self.X = np.empty((0, 5))
        if self.Y is None:
            self.Y = np.empty((0, 2))
        if self.C is None:
            self.C = np.empty((0, 1))


@dataclass
class EvaluationResult:
    """Result from evaluating a parameter set."""
    
    lap_time: float
    tracking_error: float
    safety_margin: float  # Not used, kept for compatibility
    feasible: bool = True  # Always true - no safety constraints
    
    def __post_init__(self):
        """Validate result - always feasible now."""
        # All results are considered feasible without safety constraints
        self.feasible = True
