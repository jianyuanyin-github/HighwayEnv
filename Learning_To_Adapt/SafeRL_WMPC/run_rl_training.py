#!/usr/bin/env python3
"""
Run reinforcement learning training for MPCC parameter tuning.
"""

import os
import sys
import numpy as np
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.train_mpcc_rl import MPCCRLTrainer
from Learning_To_Adapt.SafeRL_WMPC.helpers import load_config


def main():
    """Main function to run RL training."""
    # Change to project root directory
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    print("=" * 60)
    print("MPCC Reinforcement Learning Training")
    print("=" * 60)
    
    # Configuration file
    config_path = "Learning_To_Adapt/SafeRL_WMPC/_config/rl_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return False
    
    try:
        # Create trainer
        print("Initializing RL trainer...")
        trainer = MPCCRLTrainer(config_path)
        
        # Train model
        print("Starting training...")
        trainer.train()
        
        print("\nTraining completed!")
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ RL training completed successfully!")
    else:
        print("\n✗ RL training failed!")
        sys.exit(1)