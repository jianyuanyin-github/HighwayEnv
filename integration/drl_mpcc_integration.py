#!/usr/bin/env python3
"""
Safety-Aware DRL-MPCC Integration Framework

This module implements the feedforward integration of Deep Reinforcement Learning
with Model Predictive Contouring Control as described in the paper.

Architecture:
    DRL Agent → Reference Control (a_ref, δ_ref) → MPCC → Safe Control Execution
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import yaml


class SafetyAwareDRLMPCC:
    """
    Safety-aware integration of DRL policy with MPCC controller.

    The DRL agent provides feedforward reference control (a_ref, δ_ref),
    and MPCC optimizes the actual control under safety constraints with
    time-decaying weight ω(t) = exp(-α·t).
    """

    def __init__(
        self,
        mpcc_controller,
        drl_policy=None,
        alpha: float = 0.1,
        beta: float = 1.0,
        K_a: float = 100.0,
        K_delta: float = 100.0,
    ):
        """
        Initialize the integrated controller.

        Args:
            mpcc_controller: MPCC controller instance
            drl_policy: DRL policy network (can be None, will be set later)
            alpha: Time decay rate for ω(t) = exp(-α·t_k)
            beta: Overall weight for reference tracking term
            K_a: Weight for acceleration tracking
            K_delta: Weight for steering tracking
        """
        self.mpcc = mpcc_controller
        self.drl_policy = drl_policy

        # Integration parameters from paper
        self.alpha = alpha
        self.beta = beta
        self.K_a = K_a
        self.K_delta = K_delta

        # Store reference history for logging
        self.u_ref_history = []
        self.u_actual_history = []

    def set_drl_policy(self, policy):
        """Set the DRL policy after initialization."""
        self.drl_policy = policy

    def compute_omega(self, t: float) -> float:
        """
        Compute time-decaying weight ω(t) = exp(-α·t).

        Args:
            t: Time from current step (in seconds)

        Returns:
            Weight value in [0, 1]
        """
        return np.exp(-self.alpha * t)

    def get_drl_reference(self, obs: np.ndarray) -> Tuple[float, float]:
        """
        Query DRL policy for reference control.

        Args:
            obs: Current observation/state

        Returns:
            (a_ref, delta_ref): Reference acceleration [m/s²] and steering angle [rad]
        """
        if self.drl_policy is None:
            # No DRL policy available, return zero reference
            return 0.0, 0.0

        # Get action from DRL policy (normalized to [-1, 1])
        # PPO trained with highway_env outputs actions in normalized space
        action, _ = self.drl_policy.predict(obs, deterministic=True)

        # Map from normalized action space to physical units
        # highway_env's ContinuousAction uses:
        #   - ACCELERATION_RANGE = (-5, 5) m/s²
        #   - STEERING_RANGE = (-π/4, π/4) rad
        a_ref = 5.0 * float(action[0])              # [-1,1] → [-5,5] m/s²
        delta_ref = (np.pi / 4) * float(action[1])  # [-1,1] → [-π/4,π/4] rad

        return a_ref, delta_ref

    def compute_reference_weights(self) -> np.ndarray:
        """
        Compute time-decaying weights for each stage in MPCC horizon.

        Returns:
            weights: Array of shape (N,) containing ω(t_k) for each stage
        """
        N = self.mpcc.N
        Tf = self.mpcc.Tf
        dt = Tf / N

        # Time at each stage: t_k = k * dt
        times = np.arange(N) * dt

        # ω(t_k) = exp(-α·t_k)
        weights = np.exp(-self.alpha * times)

        return weights

    def update_mpcc_with_reference(
        self,
        u_ref: Tuple[float, float],
        enemyinfo: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Update MPCC controller with DRL reference.

        This method modifies MPCC's cost parameters to incorporate the
        reference tracking term J_ref.

        Args:
            u_ref: (a_ref, delta_ref) from DRL policy
            enemyinfo: Obstacle information

        Returns:
            z_current: Optimized trajectory from MPCC
        """
        a_ref, delta_ref = u_ref

        # Compute time-decaying weights
        omega_weights = self.compute_reference_weights()

        # Store reference for logging
        self.u_ref_history.append([a_ref, delta_ref])

        # Update MPCC parameters with reference tracking weights
        # This will be used in the modified MPCC cost function
        self.mpcc.u_ref = (a_ref, delta_ref)
        self.mpcc.omega_weights = omega_weights
        self.mpcc.K_ref = np.array([self.K_a, self.K_delta])
        self.mpcc.beta = self.beta

        # Call MPCC update (with modified cost function)
        z_current = self.mpcc.update(enemyinfo)

        # Log actual control
        u_actual = z_current[0, :3]  # [jerk, deltadot, thetadot]
        a_actual = z_current[0, self.mpcc.zvars.index("a")]
        delta_actual = z_current[0, self.mpcc.zvars.index("delta")]
        self.u_actual_history.append([a_actual, delta_actual])

        return z_current

    def step(
        self,
        obs: np.ndarray,
        enemyinfo: Optional[Dict] = None,
        use_drl: bool = True
    ) -> Dict[str, Any]:
        """
        Execute one control step with DRL-MPCC integration.

        Args:
            obs: Current observation for DRL policy
            enemyinfo: Obstacle information
            use_drl: Whether to use DRL reference (if False, MPCC runs standalone)

        Returns:
            result: Dictionary containing:
                - z_current: MPCC trajectory
                - u_ref: DRL reference control
                - u_actual: Actual control executed
                - omega_weights: Time-decaying weights used
        """
        # Get DRL reference
        if use_drl and self.drl_policy is not None:
            u_ref = self.get_drl_reference(obs)
        else:
            u_ref = (0.0, 0.0)  # Zero reference = pure MPCC

        # Update MPCC with reference
        z_current = self.update_mpcc_with_reference(u_ref, enemyinfo)

        # Extract actual control
        a_actual = z_current[0, self.mpcc.zvars.index("a")]
        delta_actual = z_current[0, self.mpcc.zvars.index("delta")]
        u_actual = (a_actual, delta_actual)

        return {
            "z_current": z_current,
            "u_ref": u_ref,
            "u_actual": u_actual,
            "omega_weights": self.mpcc.omega_weights if hasattr(self.mpcc, 'omega_weights') else None
        }

    def get_tracking_error(self) -> Dict[str, float]:
        """
        Compute tracking error statistics between reference and actual control.

        Returns:
            stats: Dictionary with RMSE and max errors
        """
        if len(self.u_ref_history) == 0:
            return {"rmse_a": 0.0, "rmse_delta": 0.0, "max_error_a": 0.0, "max_error_delta": 0.0}

        u_ref = np.array(self.u_ref_history)
        u_actual = np.array(self.u_actual_history)

        errors = u_ref - u_actual

        rmse_a = np.sqrt(np.mean(errors[:, 0]**2))
        rmse_delta = np.sqrt(np.mean(errors[:, 1]**2))
        max_error_a = np.max(np.abs(errors[:, 0]))
        max_error_delta = np.max(np.abs(errors[:, 1]))

        return {
            "rmse_a": rmse_a,
            "rmse_delta": rmse_delta,
            "max_error_a": max_error_a,
            "max_error_delta": max_error_delta
        }

    def reset_history(self):
        """Clear tracking history."""
        self.u_ref_history = []
        self.u_actual_history = []


def load_integration_config(config_path: str) -> Dict:
    """
    Load integration configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def create_integrated_controller(
    mpcc_controller,
    drl_policy=None,
    config_path: Optional[str] = None
) -> SafetyAwareDRLMPCC:
    """
    Factory function to create integrated controller.

    Args:
        mpcc_controller: MPCC controller instance
        drl_policy: DRL policy (optional)
        config_path: Path to integration config (optional)

    Returns:
        Integrated controller instance
    """
    if config_path is not None:
        config = load_integration_config(config_path)
        integration_params = config.get('integration', {})
    else:
        integration_params = {}

    return SafetyAwareDRLMPCC(
        mpcc_controller=mpcc_controller,
        drl_policy=drl_policy,
        alpha=integration_params.get('alpha', 0.1),
        beta=integration_params.get('beta', 1.0),
        K_a=integration_params.get('K_a', 100.0),
        K_delta=integration_params.get('K_delta', 100.0)
    )
