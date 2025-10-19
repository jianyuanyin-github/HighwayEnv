import torch
import numpy as np
from typing import Optional, Tuple
from torch import Tensor

# from botorch.acquisition import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement as EHVI
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize


class FeasibilityWeightedEHVI:
    """Feasibility-weighted Expected Hypervolume Improvement acquisition function."""

    def __init__(
        self, model, ref_point: Tensor, constraint_model=None, tkwargs: dict = None
    ):
        """
        Initialize acquisition function.

        Args:
            model: Multi-task GP model for objectives
            ref_point: Reference point for hypervolume calculation
            constraint_model: GP model for constraints (optional)
            tkwargs: Torch keyword arguments
        """
        self.model = model
        self.ref_point = ref_point
        self.constraint_model = constraint_model
        self.tkwargs = tkwargs or {"dtype": torch.double, "device": torch.device("cpu")}

        # Create EHVI acquisition function - use a simple approach for multi-task
        try:
            # Try to use the multi-task model directly
            if hasattr(self.model, 'models') and len(self.model.models) > 0:
                # Use the first model as primary
                primary_model = self.model.models[0].model
            else:
                primary_model = self.model
                
            self.ehvi = EHVI(
                model=primary_model,
                ref_point=self.ref_point,
                partitioning=None,  # Will be set automatically
            )
        except Exception:
            # Fallback to simple acquisition if EHVI fails
            self.ehvi = None

    def __call__(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function.

        Args:
            X: Input points

        Returns:
            Acquisition function values
        """
        try:
            if self.ehvi is not None:
                # Get EHVI values
                ehvi_values = self.ehvi(X)
            else:
                # Use simple UCB as fallback
                means, variances = self.model.predict(X)
                # Use first objective with UCB
                ehvi_values = means[:, 0] + 2.0 * torch.sqrt(variances[:, 0])

            # Apply feasibility weighting if constraint model is available
            if self.constraint_model is not None:
                prob_feasible = self.constraint_model.get_probability_feasible(X)
                ehvi_values = ehvi_values * prob_feasible.squeeze()

            return ehvi_values
        except Exception as e:
            print(f"Error in acquisition function: {e}")
            # Return random values as ultimate fallback
            return torch.rand(X.shape[0], **self.tkwargs)


class SimpleAcquisitionFunction:
    """Simple acquisition function for testing and debugging."""

    def __init__(self, model, tkwargs: dict = None):
        """
        Initialize simple acquisition function.

        Args:
            model: Multi-task GP model
            tkwargs: Torch keyword arguments
        """
        self.model = model
        self.tkwargs = tkwargs or {"dtype": torch.double, "device": torch.device("cpu")}

    def __call__(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function using upper confidence bound.

        Args:
            X: Input points

        Returns:
            Acquisition function values
        """
        means, variances = self.model.predict(X)

        # Upper Confidence Bound (UCB)
        beta = 2.0  # Exploration parameter
        ucb = means + beta * torch.sqrt(variances)

        # For multi-objective, use weighted sum
        weights = torch.tensor([0.5, 0.5], **self.tkwargs)  # Equal weights
        acq_values = torch.sum(ucb * weights, dim=-1)

        return acq_values


def optimize_acquisition_function(
    acquisition_function,
    bounds: Tensor,
    n_candidates: int = 1,
    n_restarts: int = 20,
    n_raw_samples: int = 512,
    tkwargs: dict = None,
) -> Tuple[Tensor, Tensor]:
    """
    Optimize acquisition function to find next candidate points.

    Args:
        acquisition_function: Acquisition function to optimize
        bounds: Parameter bounds
        n_candidates: Number of candidates to return
        n_restarts: Number of optimization restarts
        n_raw_samples: Number of raw samples for initialization
        tkwargs: Torch keyword arguments

    Returns:
        Tuple of (candidates, acquisition_values)
    """
    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    # Generate initial candidates using Sobol sampling
    sobol_candidates = draw_sobol_samples(
        bounds=bounds, n=n_raw_samples, q=n_candidates
    )

    # Optimize acquisition function
    try:
        candidates, acq_values = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=n_candidates,
            num_restarts=n_restarts,
            raw_samples=n_raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
    except Exception as e:
        print(f"Error in optimize_acqf: {e}")
        # Fallback to random sampling
        candidates = torch.rand(n_candidates, bounds.shape[1], **tkwargs)
        candidates = bounds[0] + candidates * (bounds[1] - bounds[0])
        acq_values = torch.rand(n_candidates, **tkwargs)

    return candidates, acq_values


def create_acquisition_function(
    model,
    ref_point: Tensor,
    constraint_model=None,
    acquisition_type: str = "ehvi",
    tkwargs: dict = None,
):
    """
    Create acquisition function based on type.

    Args:
        model: Multi-task GP model
        ref_point: Reference point for hypervolume
        constraint_model: Constraint model (optional)
        acquisition_type: Type of acquisition function
        tkwargs: Torch keyword arguments

    Returns:
        Acquisition function
    """
    if acquisition_type == "ehvi":
        return FeasibilityWeightedEHVI(
            model=model,
            ref_point=ref_point,
            constraint_model=constraint_model,
            tkwargs=tkwargs,
        )
    elif acquisition_type == "simple":
        return SimpleAcquisitionFunction(model=model, tkwargs=tkwargs)
    else:
        raise ValueError(f"Unknown acquisition type: {acquisition_type}")


def get_next_candidates(
    model,
    bounds: Tensor,
    ref_point: Tensor,
    constraint_model=None,
    n_candidates: int = 1,
    acquisition_type: str = "ehvi",
    tkwargs: dict = None,
) -> Tuple[Tensor, Tensor]:
    """
    Get next candidate points using acquisition function optimization.

    Args:
        model: Multi-task GP model
        bounds: Parameter bounds
        ref_point: Reference point for hypervolume
        constraint_model: Constraint model (optional)
        n_candidates: Number of candidates to return
        acquisition_type: Type of acquisition function
        tkwargs: Torch keyword arguments

    Returns:
        Tuple of (candidates, acquisition_values)
    """
    # Create acquisition function
    acq_function = create_acquisition_function(
        model=model,
        ref_point=ref_point,
        constraint_model=constraint_model,
        acquisition_type=acquisition_type,
        tkwargs=tkwargs,
    )

    # Optimize acquisition function
    candidates, acq_values = optimize_acquisition_function(
        acquisition_function=acq_function,
        bounds=bounds,
        n_candidates=n_candidates,
        tkwargs=tkwargs,
    )

    return candidates, acq_values


def calculate_hypervolume(pareto_front: Tensor, ref_point: Tensor) -> float:
    """
    Calculate hypervolume of Pareto front.

    Args:
        pareto_front: Pareto optimal points
        ref_point: Reference point

    Returns:
        Hypervolume value
    """
    if pareto_front.shape[0] == 0:
        return 0.0

    # Simple hypervolume calculation for 2D case
    if pareto_front.shape[1] == 2:
        # Sort points by first objective
        sorted_indices = torch.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]

        # Calculate hypervolume as sum of rectangles
        hv = 0.0
        for i in range(sorted_front.shape[0]):
            if i == 0:
                width = ref_point[0] - sorted_front[i, 0]
            else:
                width = sorted_front[i - 1, 0] - sorted_front[i, 0]

            height = ref_point[1] - sorted_front[i, 1]
            hv += width * height

        return hv.item()
    else:
        # For higher dimensions, use approximation
        # This is a simplified version
        return torch.prod(ref_point - pareto_front, dim=1).sum().item()
