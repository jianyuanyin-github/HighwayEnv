#!/usr/bin/env python3
"""
Test script for Bayesian Optimization module.
This script tests the basic functionality without requiring the full MPCC simulation.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BO_WMPC.bayesian_optimization import BayesianOptimizer
from BO_WMPC.objective_function import ObjectiveResult
from helpers import load_config


def create_mock_objective_function():
    """Create a mock objective function for testing."""

    def mock_evaluate_objective(params, segments):
        """Mock objective function that returns realistic values."""
        # Extract parameters
        q_l = params.get("q_l", 0.1)
        q_c = params.get("q_c", 0.1)
        q_v = params.get("q_v", 0.1)
        q_theta = params.get("q_theta", 0.1)

        # Create mock objectives based on parameters
        # Simulate lap time (lower is better)
        lap_time = (
            10.0 + 5.0 * (1.0 - q_l) + 3.0 * (1.0 - q_c) + np.random.normal(0, 0.1)
        )

        # Simulate tracking error (lower is better)
        tracking_error = (
            0.5 + 2.0 * (1.0 - q_v) + 1.5 * (1.0 - q_theta) + np.random.normal(0, 0.05)
        )

        # Simulate safety margin (higher is better, should be > 0 for feasibility)
        safety_margin = (
            0.2 + 0.3 * q_l + 0.2 * q_c - 0.1 * q_v + np.random.normal(0, 0.02)
        )

        return ObjectiveResult(
            lap_time=lap_time,
            tracking_error=tracking_error,
            safety_margin=safety_margin,
            success=True,
        )

    return mock_evaluate_objective


def test_bo_initialization():
    """Test Bayesian Optimizer initialization."""
    print("Testing BO initialization...")

    try:
        # Create optimizer
        optimizer = BayesianOptimizer()
        print("✓ BO initialization successful")

        # Test bounds setup
        print(f"Parameter bounds shape: {optimizer.bounds.shape}")
        print(f"Reference point: {optimizer.ref_point}")

        return optimizer

    except Exception as e:
        print(f"✗ BO initialization failed: {e}")
        return None


def test_bo_initialization_with_mock():
    """Test BO initialization with mock objective function."""
    print("\nTesting BO with mock objective function...")

    try:
        # Create optimizer
        optimizer = BayesianOptimizer()

        # Replace objective function with mock
        import BO_WMPC.objective_function

        BO_WMPC.objective_function.evaluate_objective = create_mock_objective_function()

        # Initialize with random points
        optimizer.initialize_random(n_init=3)
        print("✓ BO initialization with mock function successful")

        return optimizer

    except Exception as e:
        print(f"✗ BO initialization with mock failed: {e}")
        return None


def test_model_fitting(optimizer):
    """Test surrogate model fitting."""
    print("\nTesting model fitting...")

    try:
        optimizer.fit_models()
        print("✓ Model fitting successful")

        # Test prediction
        test_X = torch.randn(5, optimizer.bounds.shape[1], **optimizer.tkwargs)
        means, variances = optimizer.objective_model.predict(test_X)
        print(f"✓ Prediction successful - means shape: {means.shape}")

        return True

    except Exception as e:
        print(f"✗ Model fitting failed: {e}")
        return False


def test_acquisition_function(optimizer):
    """Test acquisition function optimization."""
    print("\nTesting acquisition function...")

    try:
        candidates, acq_values = optimizer.get_next_candidates(n_candidates=2)
        print(
            f"✓ Acquisition function successful - candidates shape: {candidates.shape}"
        )
        print(f"Candidates: {candidates.cpu().numpy()}")
        print(f"Acquisition values: {acq_values.cpu().numpy()}")

        return True

    except Exception as e:
        print(f"✗ Acquisition function failed: {e}")
        return False


def test_full_optimization_loop(optimizer):
    """Test a few iterations of the optimization loop."""
    print("\nTesting full optimization loop...")

    try:
        # Run a few iterations
        results = optimizer.optimize(n_iterations=3)
        print("✓ Full optimization loop successful")

        # Print results
        print(f"Total evaluations: {len(results.Y)}")
        print(f"Best lap time: {np.min(results.Y[:, 0]):.3f}")
        print(f"Best tracking error: {np.min(results.Y[:, 1]):.3f}")

        # Get best parameters
        best_params = optimizer.get_best_parameters()
        print(f"Best parameters: {best_params}")

        return True

    except Exception as e:
        print(f"✗ Full optimization loop failed: {e}")
        return False


def test_plotting(optimizer):
    """Test result plotting."""
    print("\nTesting result plotting...")

    try:
        optimizer.plot_results("test_bo_plot.png")
        print("✓ Result plotting successful")

        # Check if file was created
        if os.path.exists("test_bo_plot.png"):
            print("✓ Plot file created successfully")
            return True
        else:
            print("✗ Plot file not created")
            return False

    except Exception as e:
        print(f"✗ Result plotting failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 50)
    print("Testing Bayesian Optimization Module")
    print("=" * 50)

    # Test 1: Basic initialization
    optimizer = test_bo_initialization()
    if optimizer is None:
        print("Basic initialization failed. Exiting.")
        return

    # Test 2: Initialization with mock function
    optimizer = test_bo_initialization_with_mock()
    if optimizer is None:
        print("Mock initialization failed. Exiting.")
        return

    # Test 3: Model fitting
    if not test_model_fitting(optimizer):
        print("Model fitting failed. Exiting.")
        return

    # Test 4: Acquisition function
    if not test_acquisition_function(optimizer):
        print("Acquisition function failed. Exiting.")
        return

    # Test 5: Full optimization loop
    if not test_full_optimization_loop(optimizer):
        print("Full optimization loop failed. Exiting.")
        return

    # Test 6: Plotting
    test_plotting(optimizer)

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
