import torch
import numpy as np
import yaml
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from torch import Tensor
import matplotlib.pyplot as plt
from pathlib import Path

from .surrogate_models import create_surrogate_models, MultiTaskGPModel, ConstraintModel
from .acquisition import get_next_candidates, calculate_hypervolume
from .objective_function import evaluate_objective
from .track_segmentation import get_train_segments
from .dataclasses import BOResults, OptimizationState
from ..helpers import load_config, setup_logging, save_results


class BayesianOptimizer:
    """Main Bayesian Optimization class for MPCC parameter tuning."""

    def __init__(self, config_path: str = None):
        """
        Initialize Bayesian Optimizer.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "_config", "bo_config.yaml"
            )

        self.config = load_config(config_path)
        self.logger = setup_logging("BO_Optimizer")

        # Setup torch
        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device(self.config.get("device", "cpu")),
        }

        # Initialize optimization state
        self.state = OptimizationState()

        # Parameter bounds
        self.bounds = self._setup_bounds()

        # Reference point for hypervolume calculation
        self.ref_point = torch.tensor(self.config["reference_point"], **self.tkwargs)

        # Initialize models
        self.objective_model = None
        self.constraint_model = None

        # Results storage
        self.results = BOResults()

        # Track segments
        self.segments = get_train_segments()

        self.logger.info("Bayesian Optimizer initialized successfully")

    def _setup_bounds(self) -> Tensor:
        """Setup parameter bounds from configuration."""
        bounds_config = self.config["parameter_bounds"]
        bounds_list = []

        for param_name in self.config["parameters"]:
            if param_name in bounds_config:
                bounds_list.append(bounds_config[param_name])
            else:
                # Default bounds if not specified
                bounds_list.append([0.0, 1.0])

        return torch.tensor(bounds_list, **self.tkwargs).T

    def initialize_random(self, n_init: int = None) -> None:
        """
        Initialize optimization with random points.

        Args:
            n_init: Number of initial points (default from config)
        """
        if n_init is None:
            n_init = self.config.get("n_initial", 5)

        self.logger.info(f"Initializing with {n_init} random points")

        # Generate random initial points
        from botorch.utils.sampling import draw_sobol_samples

        X_init = draw_sobol_samples(bounds=self.bounds, n=n_init, q=1).squeeze(1)

        # Evaluate initial points
        Y_init = []
        C_init = []

        for i, x in enumerate(X_init):
            self.logger.info(f"Evaluating initial point {i+1}/{n_init}")

            # Convert to parameter dictionary
            params = self._tensor_to_params(x)

            # Evaluate objective
            try:
                result = evaluate_objective(params, self.segments)
                
                # Check for invalid results and convert to finite values
                lap_time = result.lap_time if result.lap_time != float('inf') else 100.0
                tracking_error = result.tracking_error if result.tracking_error != float('inf') else 10.0
                
                # Ensure finite values
                lap_time = max(0.1, min(lap_time, 100.0))
                tracking_error = max(0.01, min(tracking_error, 10.0))
                
                Y_init.append([lap_time, tracking_error])
                C_init.append([1.0])  # Always feasible - no safety constraint

                self.logger.info(
                    f"Point {i+1}: lap_time={lap_time:.3f}, "
                    f"tracking_error={tracking_error:.3f}"
                )

            except Exception as e:
                self.logger.error(f"Error evaluating point {i+1}: {e}")
                # Use reasonable default values instead of inf
                Y_init.append([50.0, 5.0])  # High but finite penalty values
                C_init.append([1.0])  # Always feasible

        # Convert to tensors
        self.state.train_X = X_init
        self.state.train_Y = torch.tensor(Y_init, **self.tkwargs)
        self.state.train_C = torch.tensor(C_init, **self.tkwargs)

        # Update results
        self.results.X = X_init.cpu().numpy()
        self.results.Y = self.state.train_Y.cpu().numpy()
        self.results.C = self.state.train_C.cpu().numpy()

        self.logger.info("Initialization completed")

    def _tensor_to_params(self, x: Tensor) -> Dict[str, float]:
        """Convert tensor to parameter dictionary."""
        params = {}
        for i, param_name in enumerate(self.config["parameters"]):
            params[param_name] = x[i].item()
        return params

    def _params_to_tensor(self, params: Dict[str, float]) -> Tensor:
        """Convert parameter dictionary to tensor."""
        x = []
        for param_name in self.config["parameters"]:
            x.append(params[param_name])
        return torch.tensor(x, **self.tkwargs)

    def fit_models(self) -> None:
        """Fit surrogate models to current data."""
        if self.state.train_X.shape[0] == 0:
            raise ValueError(
                "No training data available. Call initialize_random() first."
            )

        self.logger.info("Fitting surrogate models")

        # Normalize inputs
        from botorch.utils.transforms import normalize

        X_normalized = normalize(self.state.train_X, self.bounds)

        # Create models
        self.objective_model, self.constraint_model = create_surrogate_models(
            train_X=X_normalized,
            train_Y=self.state.train_Y,
            train_C=self.state.train_C,
            tkwargs=self.tkwargs,
        )

        self.logger.info("Models fitted successfully")

    def get_next_candidates(self, n_candidates: int = 1) -> Tuple[Tensor, Tensor]:
        """
        Get next candidate points using acquisition function.

        Args:
            n_candidates: Number of candidates to return

        Returns:
            Tuple of (candidates, acquisition_values)
        """
        if self.objective_model is None:
            raise ValueError("Models not fitted. Call fit_models() first.")

        # Normalize bounds for optimization
        from botorch.utils.transforms import normalize

        normalized_bounds = torch.zeros(2, self.bounds.shape[1], **self.tkwargs)
        normalized_bounds[1] = 1.0

        # Get candidates
        candidates, acq_values = get_next_candidates(
            model=self.objective_model,
            bounds=normalized_bounds,
            ref_point=self.ref_point,
            constraint_model=self.constraint_model,
            n_candidates=n_candidates,
            acquisition_type=self.config.get("acquisition_type", "ehvi"),
            tkwargs=self.tkwargs,
        )

        # Unnormalize candidates
        from botorch.utils.transforms import unnormalize

        candidates = unnormalize(candidates, self.bounds)

        return candidates, acq_values

    def evaluate_candidates(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Evaluate candidate points.

        Args:
            candidates: Candidate points to evaluate

        Returns:
            Tuple of (objectives, constraints)
        """
        n_candidates = candidates.shape[0]
        Y_candidates = []
        C_candidates = []

        for i in range(n_candidates):
            self.logger.info(f"Evaluating candidate {i+1}/{n_candidates}")

            # Convert to parameter dictionary
            params = self._tensor_to_params(candidates[i])

            # Evaluate objective
            try:
                result = evaluate_objective(params, self.segments)
                
                # Check for invalid results and convert to finite values
                lap_time = result.lap_time if result.lap_time != float('inf') else 100.0
                tracking_error = result.tracking_error if result.tracking_error != float('inf') else 10.0
                
                # Ensure finite values
                lap_time = max(0.1, min(lap_time, 100.0))
                tracking_error = max(0.01, min(tracking_error, 10.0))
                
                Y_candidates.append([lap_time, tracking_error])
                C_candidates.append([1.0])  # Always feasible - no safety constraint

                self.logger.info(
                    f"Candidate {i+1}: lap_time={lap_time:.3f}, "
                    f"tracking_error={tracking_error:.3f}"
                )

            except Exception as e:
                self.logger.error(f"Error evaluating candidate {i+1}: {e}")
                # Use reasonable default values instead of inf
                Y_candidates.append([50.0, 5.0])  # High but finite penalty values
                C_candidates.append([1.0])  # Always feasible

        return torch.tensor(Y_candidates, **self.tkwargs), torch.tensor(
            C_candidates, **self.tkwargs
        )

    def update_data(self, new_X: Tensor, new_Y: Tensor, new_C: Tensor) -> None:
        """
        Update training data with new observations.

        Args:
            new_X: New input points
            new_Y: New objective values
            new_C: New constraint values
        """
        # Concatenate new data
        self.state.train_X = torch.cat([self.state.train_X, new_X], dim=0)
        self.state.train_Y = torch.cat([self.state.train_Y, new_Y], dim=0)
        self.state.train_C = torch.cat([self.state.train_C, new_C], dim=0)

        # Update results
        self.results.X = self.state.train_X.cpu().numpy()
        self.results.Y = self.state.train_Y.cpu().numpy()
        self.results.C = self.state.train_C.cpu().numpy()

        self.logger.info(
            f"Data updated. Total observations: {self.state.train_X.shape[0]}"
        )

    def optimize(self, n_iterations: int = None) -> BOResults:
        """
        Run Bayesian optimization.

        Args:
            n_iterations: Number of optimization iterations (default from config)

        Returns:
            Optimization results
        """
        if n_iterations is None:
            n_iterations = self.config.get("n_bayesian_optimization", 20)

        self.logger.info(
            f"Starting Bayesian optimization with {n_iterations} iterations"
        )

        # Initialize if no data
        if self.state.train_X.shape[0] == 0:
            self.initialize_random()

        # Main optimization loop
        for iteration in range(n_iterations):
            self.logger.info(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

            # Fit models
            self.fit_models()

            # Get next candidates
            candidates, acq_values = self.get_next_candidates(
                n_candidates=self.config.get("n_candidates_per_iteration", 1)
            )

            self.logger.info(f"Selected candidates: {candidates.cpu().numpy()}")
            self.logger.info(f"Acquisition values: {acq_values.cpu().numpy()}")

            # Evaluate candidates
            Y_new, C_new = self.evaluate_candidates(candidates)

            # Update data
            self.update_data(candidates, Y_new, C_new)

            # Save intermediate results
            if (iteration + 1) % self.config.get("save_interval", 5) == 0:
                self.save_results(f"bo_results_iter_{iteration + 1}.npz")

            # Check convergence
            if self._check_convergence():
                self.logger.info("Convergence criteria met. Stopping optimization.")
                break

        # Final save
        self.save_results("bo_results_final.npz")
        
        # Save Pareto front
        self.save_pareto_front("pareto_front_final.npz")
        
        # Print Pareto front analysis
        pareto_X, pareto_Y = self.get_pareto_front()
        if len(pareto_X) > 0:
            self.logger.info(f"\n=== PARETO FRONT ANALYSIS ===")
            self.logger.info(f"Number of Pareto optimal solutions: {len(pareto_X)}")
            self.logger.info(f"Hypervolume: {self.calculate_hypervolume(pareto_Y):.6f}")
            
            # Show best solutions for each objective
            best_time_idx = np.argmin(pareto_Y[:, 0])
            best_error_idx = np.argmin(pareto_Y[:, 1])
            
            self.logger.info(f"\nBest lap time solution:")
            best_time_params = self._tensor_to_params(torch.tensor(pareto_X[best_time_idx], **self.tkwargs))
            self.logger.info(f"  Parameters: {best_time_params}")
            self.logger.info(f"  Lap time: {pareto_Y[best_time_idx, 0]:.3f}s")
            self.logger.info(f"  Tracking error: {pareto_Y[best_time_idx, 1]:.3f}m")
            
            self.logger.info(f"\nBest tracking error solution:")
            best_error_params = self._tensor_to_params(torch.tensor(pareto_X[best_error_idx], **self.tkwargs))
            self.logger.info(f"  Parameters: {best_error_params}")
            self.logger.info(f"  Lap time: {pareto_Y[best_error_idx, 0]:.3f}s")
            self.logger.info(f"  Tracking error: {pareto_Y[best_error_idx, 1]:.3f}m")

        self.logger.info("Bayesian optimization completed")
        return self.results

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if self.state.train_X.shape[0] < 10:
            return False

        # Check if best objective has improved in last 5 iterations
        recent_Y = self.state.train_Y[-5:]
        best_recent = torch.min(recent_Y[:, 0])  # Assuming first objective is lap time

        if self.state.train_X.shape[0] >= 10:
            previous_Y = self.state.train_Y[-10:-5]
            best_previous = torch.min(previous_Y[:, 0])

            improvement = best_previous - best_recent
            if improvement < self.config.get("convergence_threshold", 0.01):
                return True

        return False

    def save_results(self, filename: str) -> None:
        """Save optimization results."""
        save_path = os.path.join(self.config.get("results_dir", "results"), filename)
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_results(self.results, save_path)
        self.logger.info(f"Results saved to {save_path}")

    def plot_results(self, save_path: str = None) -> None:
        """Plot optimization results."""
        if save_path is None:
            save_path = os.path.join(
                self.config.get("results_dir", "results"), "bo_plot.png"
            )

        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Objective values over iterations
        axes[0, 0].plot(self.results.Y[:, 0], "b-", label="Lap Time")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Lap Time (s)")
        axes[0, 0].set_title("Lap Time vs Iteration")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.results.Y[:, 1], "r-", label="Tracking Error")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Tracking Error (m)")
        axes[0, 1].set_title("Tracking Error vs Iteration")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 2: Pareto front
        axes[1, 0].scatter(self.results.Y[:, 0], self.results.Y[:, 1], alpha=0.6, label="All points")
        
        # Highlight Pareto front
        pareto_X, pareto_Y = self.get_pareto_front()
        if len(pareto_Y) > 0:
            axes[1, 0].scatter(pareto_Y[:, 0], pareto_Y[:, 1], color='red', s=50, label="Pareto front")
            
            # Connect Pareto points for visualization
            sorted_indices = np.argsort(pareto_Y[:, 0])
            sorted_pareto = pareto_Y[sorted_indices]
            axes[1, 0].plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r--', alpha=0.5)
        
        axes[1, 0].set_xlabel("Lap Time (s)")
        axes[1, 0].set_ylabel("Tracking Error (m)")
        axes[1, 0].set_title("Pareto Front")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot 3: Constraint values
        if self.results.C is not None:
            axes[1, 1].plot(self.results.C[:, 0], "g-", label="Safety Margin")
            axes[1, 1].axhline(
                y=0, color="r", linestyle="--", label="Constraint Boundary"
            )
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Safety Margin")
            axes[1, 1].set_title("Constraint Values")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Results plot saved to {save_path}")

    def get_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Pareto optimal points from optimization results.
        
        Returns:
            Tuple of (pareto_X, pareto_Y) - parameter sets and objective values on Pareto front
        """
        if self.results.Y is None or len(self.results.Y) == 0:
            return np.empty((0, 5)), np.empty((0, 2))
        
        # Consider all points (no feasibility constraint)
        all_X = self.results.X
        all_Y = self.results.Y
        
        if len(all_Y) == 0:
            return np.empty((0, 5)), np.empty((0, 2))
        
        # Find Pareto optimal points
        pareto_mask = self._get_pareto_mask(all_Y)
        pareto_X = all_X[pareto_mask]
        pareto_Y = all_Y[pareto_mask]
        
        self.logger.info(f"Found {len(pareto_X)} points on Pareto front")
        
        return pareto_X, pareto_Y
    
    def _get_pareto_mask(self, Y: np.ndarray) -> np.ndarray:
        """Get boolean mask for Pareto optimal points."""
        if len(Y) == 0:
            return np.empty(0, dtype=bool)
        
        # For minimization problem, find non-dominated points
        n_points = len(Y)
        pareto_mask = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                        pareto_mask[i] = False
                        break
        
        return pareto_mask
    
    def calculate_hypervolume(self, pareto_Y: np.ndarray = None) -> float:
        """
        Calculate hypervolume of Pareto front.
        
        Args:
            pareto_Y: Pareto front objective values (if None, compute from results)
            
        Returns:
            Hypervolume value
        """
        if pareto_Y is None:
            _, pareto_Y = self.get_pareto_front()
        
        if len(pareto_Y) == 0:
            return 0.0
        
        # Reference point from config
        ref_point = np.array(self.config["reference_point"])
        
        # Simple hypervolume calculation for 2D case
        if pareto_Y.shape[1] == 2:
            # Sort points by first objective
            sorted_indices = np.argsort(pareto_Y[:, 0])
            sorted_front = pareto_Y[sorted_indices]
            
            # Calculate hypervolume as sum of rectangles
            hv = 0.0
            for i in range(len(sorted_front)):
                if i == 0:
                    width = ref_point[0] - sorted_front[i, 0]
                else:
                    width = sorted_front[i-1, 0] - sorted_front[i, 0]
                
                height = ref_point[1] - sorted_front[i, 1]
                
                # Only add positive contributions
                if width > 0 and height > 0:
                    hv += width * height
            
            return hv
        else:
            # For higher dimensions, use approximation
            return np.prod(ref_point - pareto_Y, axis=1).sum()

    def get_best_parameters(self) -> Dict[str, float]:
        """Get best parameters found during optimization."""
        if self.results.Y is None or len(self.results.Y) == 0:
            raise ValueError("No optimization results available")

        # Find best point (minimum lap time among all points)
        best_idx = np.argmin(self.results.Y[:, 0])

        best_params = self._tensor_to_params(
            torch.tensor(self.results.X[best_idx], **self.tkwargs)
        )

        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best lap time: {self.results.Y[best_idx, 0]:.3f}")
        self.logger.info(f"Best tracking error: {self.results.Y[best_idx, 1]:.3f}")

        return best_params
    
    def save_pareto_front(self, filename: str = None) -> None:
        """Save Pareto front to file."""
        if filename is None:
            filename = "pareto_front.npz"
        
        pareto_X, pareto_Y = self.get_pareto_front()
        
        save_path = os.path.join(self.config.get("results_dir", "results"), filename)
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert to parameter dictionaries for easier interpretation
        pareto_params = []
        for x in pareto_X:
            params = self._tensor_to_params(torch.tensor(x, **self.tkwargs))
            pareto_params.append(params)
        
        np.savez(
            save_path,
            pareto_X=pareto_X,
            pareto_Y=pareto_Y,
            pareto_params=pareto_params,
            hypervolume=self.calculate_hypervolume(pareto_Y)
        )
        
        self.logger.info(f"Pareto front saved to {save_path}")
        self.logger.info(f"Hypervolume: {self.calculate_hypervolume(pareto_Y):.6f}")
