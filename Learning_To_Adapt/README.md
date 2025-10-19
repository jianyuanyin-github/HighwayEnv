# Adaptive Learning for RC Car MPCC

This module implements adaptive parameter tuning for the RC Car MPCC controller using a two-stage approach:
1. **Bayesian Optimization (BO)** to find Pareto optimal parameter sets
2. **Reinforcement Learning (RL)** to learn optimal parameter selection policy

## Overview

The adaptive learning system allows the MPCC controller to automatically adjust its parameters based on current driving conditions, improving performance across different track segments and scenarios.

## Directory Structure

```
Learning_To_Adapt/
├── SafeRL_WMPC/
│   ├── BO_WMPC/              # Bayesian optimization module
│   │   ├── bayesian_optimization.py
│   │   ├── objective_function.py
│   │   ├── surrogate_models.py
│   │   ├── acquisition.py
│   │   ├── dataclasses.py
│   │   ├── postprocessing.py
│   │   └── track_segmentation.py
│   ├── RL_WMPC/              # Reinforcement learning module
│   │   ├── environment.py
│   │   ├── observation.py
│   │   ├── reward.py
│   │   └── evaluation.py
│   ├── _config/              # Configuration files
│   │   ├── bo_config.yaml
│   │   └── rl_config.yaml
│   ├── _parameters/          # Pareto optimal parameter sets
│   ├── _models/              # Trained RL models
│   ├── _logs/                # Training logs
│   ├── _baseline/            # Baseline performance data
│   ├── bo_optimize.py        # BO main script
│   ├── rl_training.py        # RL training script
│   └── helpers.py            # Utility functions
└── README.md
```

## Usage

### 1. Bayesian Optimization

To find Pareto optimal parameter sets:

```bash
cd RL-and-MPC
python Learning_To_Adapt/SafeRL_WMPC/bo_optimize.py
```

This will:
- Generate initial random parameter combinations
- Perform Bayesian optimization to find Pareto optimal sets
- Save results to `_logs/` directory

### 2. Reinforcement Learning Training

To train the RL agent:

```bash
python Learning_To_Adapt/SafeRL_WMPC/rl_training.py
```

This will:
- Load Pareto optimal parameter sets
- Train PPO agent to select optimal parameters
- Save trained model to `_models/` directory

### 3. Online Adaptive Control

To use the trained model for adaptive control, modify your main simulation script:

```python
from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.environment import load_trained_model

# Load trained model
agent = load_trained_model('path/to/model')

# In control loop
observation = get_current_observation()
action = agent.predict(observation)
params = pareto_params[action]
mpcc.update_weights(params)
```

## Configuration

### BO Configuration (`_config/bo_config.yaml`)

Key parameters:
- `n_initial`: Number of random initial evaluations
- `n_bayesian_optimization`: Number of BO iterations
- `batch_size`: Parallel evaluations per iteration
- `max_lat_dev`: Maximum lateral deviation constraint
- `max_a_comb`: Maximum combined acceleration constraint

### RL Configuration (`_config/rl_config.yaml`)

Key parameters:
- `n_training_steps`: Total training steps
- `n_environments`: Number of parallel environments
- `learning_rate`: Learning rate for PPO
- `net_arch`: Neural network architecture

## Parameter Space

The system optimizes 5 MPCC parameters:
- `Qc`: Contouring error weight [100, 2000]
- `Ql`: Lag error weight [500, 3000]
- `Q_theta`: Progress reward weight [1, 50]
- `R_d`: Throttle change weight [0.001, 0.1]
- `R_delta`: Steering change weight [0.001, 0.1]

## Performance Metrics

The optimization targets two objectives:
1. **Tracking Error**: Maximum lateral deviation from reference trajectory
2. **Speed Performance**: Root mean square velocity tracking error

## Safety Constraints

The system enforces safety constraints:
- Maximum lateral deviation: 0.5m
- Maximum combined acceleration: 2.0 m/s²

## Dependencies

Required packages:
```bash
pip install torch botorch stable-baselines3 gymnasium scikit-learn matplotlib
```

## Notes

- The system is designed for RC car scale (mass ~0.041kg)
- Tracks are assumed to be in the `tracks/` directory
- Results are saved in organized directories for easy analysis
- The system supports training continuation and model loading 