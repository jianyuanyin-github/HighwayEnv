import torch
import numpy as np
from typing import Optional, Tuple
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


class ObjectiveModel:
    """Gaussian Process model for objective functions."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, tkwargs: dict):
        """
        Initialize objective model.

        Args:
            train_X: Training inputs (normalized)
            train_Y: Training outputs (objectives)
            tkwargs: Torch keyword arguments
        """
        self.tkwargs = tkwargs
        self.device = tkwargs["device"]
        self.dtype = tkwargs["dtype"]

        # Initialize model
        self.model = None
        self.mll = None

        # Fit model if data is provided
        if train_X.shape[0] > 0:
            self.fit(train_X, train_Y)

    def fit(self, train_X: Tensor, train_Y: Tensor):
        """
        Fit the Gaussian Process model.

        Args:
            train_X: Training inputs
            train_Y: Training outputs
        """
        # Create model
        self.model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=train_Y.shape[-1]),
        )

        # Create MLL
        self.mll = ExactMarginalLogLikelihood(
            likelihood=self.model.likelihood, model=self.model
        )

        # Fit model
        fit_gpytorch_mll(self.mll)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Make predictions.

        Args:
            X: Input points

        Returns:
            Tuple of (mean, variance)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean
            variance = posterior.variance

        return mean, variance

    def get_fantasy_model(self, X: Tensor, Y: Tensor) -> "ObjectiveModel":
        """
        Get fantasy model with additional data.

        Args:
            X: Additional input points
            Y: Additional output points

        Returns:
            New model with fantasy data
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        fantasy_model = self.model.get_fantasy_model(X, Y)

        # Create new model instance
        new_model = ObjectiveModel(
            train_X=torch.empty((0, X.shape[-1]), **self.tkwargs),
            train_Y=torch.empty((0, Y.shape[-1]), **self.tkwargs),
            tkwargs=self.tkwargs,
        )
        new_model.model = fantasy_model
        new_model.mll = ExactMarginalLogLikelihood(
            likelihood=fantasy_model.likelihood, model=fantasy_model
        )

        return new_model


class ConstraintModel:
    """Gaussian Process model for constraint functions."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, tkwargs: dict):
        """
        Initialize constraint model.

        Args:
            train_X: Training inputs (normalized)
            train_Y: Training outputs (constraint values)
            tkwargs: Torch keyword arguments
        """
        self.tkwargs = tkwargs
        self.device = tkwargs["device"]
        self.dtype = tkwargs["dtype"]

        # Initialize model
        self.model = None
        self.mll = None

        # Fit model if data is provided
        if train_X.shape[0] > 0:
            self.fit(train_X, train_Y)

    def fit(self, train_X: Tensor, train_Y: Tensor):
        """
        Fit the Gaussian Process model.

        Args:
            train_X: Training inputs
            train_Y: Training outputs (constraint values)
        """
        # Create model
        self.model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=train_Y.shape[-1]),
        )

        # Create MLL
        self.mll = ExactMarginalLogLikelihood(
            likelihood=self.model.likelihood, model=self.model
        )

        # Fit model
        fit_gpytorch_mll(self.mll)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Make predictions.

        Args:
            X: Input points

        Returns:
            Tuple of (mean, variance)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean
            variance = posterior.variance

        return mean, variance

    def get_probability_feasible(self, X: Tensor) -> Tensor:
        """
        Get probability that constraints are satisfied.

        Args:
            X: Input points

        Returns:
            Probability of feasibility
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        mean, variance = self.predict(X)

        # For constraint satisfaction, we want the constraint to be <= 0
        # P(constraint <= 0) = P(N(mean, var) <= 0)
        # = Phi(-mean / sqrt(var))
        std = torch.sqrt(variance)
        z_score = -mean / std

        # Use normal CDF approximation
        prob_feasible = 0.5 * (1 + torch.erf(z_score / np.sqrt(2)))

        return prob_feasible


class MultiTaskGPModel:
    """Multi-task Gaussian Process model for multiple objectives."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, tkwargs: dict):
        """
        Initialize multi-task GP model.

        Args:
            train_X: Training inputs (normalized)
            train_Y: Training outputs (multiple objectives)
            tkwargs: Torch keyword arguments
        """
        self.tkwargs = tkwargs
        self.device = tkwargs["device"]
        self.dtype = tkwargs["dtype"]

        # Initialize separate models for each objective
        self.models = []
        n_objectives = train_Y.shape[-1]

        for i in range(n_objectives):
            model = ObjectiveModel(
                train_X=train_X, train_Y=train_Y[:, i : i + 1], tkwargs=tkwargs
            )
            self.models.append(model)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Make predictions for all objectives.

        Args:
            X: Input points

        Returns:
            Tuple of (means, variances)
        """
        means = []
        variances = []

        for model in self.models:
            mean, variance = model.predict(X)
            means.append(mean)
            variances.append(variance)

        means = torch.cat(means, dim=-1)
        variances = torch.cat(variances, dim=-1)

        return means, variances

    def get_fantasy_model(self, X: Tensor, Y: Tensor) -> "MultiTaskGPModel":
        """
        Get fantasy model with additional data.

        Args:
            X: Additional input points
            Y: Additional output points

        Returns:
            New model with fantasy data
        """
        new_models = []

        for i, model in enumerate(self.models):
            fantasy_model = model.get_fantasy_model(X, Y[:, i : i + 1])
            new_models.append(fantasy_model)

        # Create new multi-task model
        new_mt_model = MultiTaskGPModel(
            train_X=torch.empty((0, X.shape[-1]), **self.tkwargs),
            train_Y=torch.empty((0, Y.shape[-1]), **self.tkwargs),
            tkwargs=self.tkwargs,
        )
        new_mt_model.models = new_models

        return new_mt_model


def create_surrogate_models(
    train_X: Tensor,
    train_Y: Tensor,
    train_C: Optional[Tensor] = None,
    tkwargs: dict = None,
) -> Tuple[MultiTaskGPModel, Optional[ConstraintModel]]:
    """
    Create surrogate models for objectives and constraints.

    Args:
        train_X: Training inputs
        train_Y: Training outputs (objectives)
        train_C: Training constraint values (optional)
        tkwargs: Torch keyword arguments

    Returns:
        Tuple of (objective_model, constraint_model)
    """
    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    # Create objective model
    objective_model = MultiTaskGPModel(train_X, train_Y, tkwargs)

    # Create constraint model if constraint data is provided
    constraint_model = None
    if train_C is not None and train_C.shape[0] > 0:
        constraint_model = ConstraintModel(train_X, train_C, tkwargs)

    return objective_model, constraint_model
