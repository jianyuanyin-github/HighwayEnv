#!/usr/bin/env python3
"""
Run Bayesian optimization for MPCC parameter tuning.
"""

import os
import sys
import numpy as np
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.bayesian_optimization import BayesianOptimizer
from Learning_To_Adapt.SafeRL_WMPC.helpers import load_config


def main():
    """Main function to run Bayesian optimization."""
    # Change to project root directory
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    print("=" * 60)
    print("MPCC Bayesian Optimization")
    print("=" * 60)
    
    # Configuration file
    config_path = "Learning_To_Adapt/SafeRL_WMPC/_config/bo_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return False
    
    try:
        # Create optimizer
        print("Initializing Bayesian optimizer...")
        optimizer = BayesianOptimizer(config_path)
        
        # Run optimization with full iterations
        print("Starting optimization...")
        result = optimizer.optimize()  # Use config file settings
        
        # Get best parameters
        print("\nOptimization completed!")
        best_params = optimizer.get_best_parameters()
        
        print(f"Best parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.4f}")
        
        # Save results
        optimizer.save_results("bo_results.npz")
        print(f"Results saved to: bo_results.npz")
        
        # Plot results
        optimizer.plot_results("bo_results.png")
        print(f"Results plot saved to: bo_results.png")
        
        return True
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ Bayesian optimization completed successfully!")
    else:
        print("\n✗ Bayesian optimization failed!")
        sys.exit(1)