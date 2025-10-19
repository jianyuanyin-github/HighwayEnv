import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MPCCObservationConfig:
    """Configuration for MPCC observation generator."""
    anticipation_horizon: int = 20  # How far ahead to look
    n_anticipation_points: int = 5  # Number of future points to include
    normalize_bounds: List[List[float]] = None  # Normalization bounds for each feature


class MPCCObservationGenerator:
    """
    Generate observations for MPCC RL environment.
    
    Observations include:
    - Current lateral deviation (normalized)
    - Current velocity error (normalized)
    - Current angular velocity (normalized)
    - Future reference trajectory information (curvature, reference velocity)
    """
    
    def __init__(self, config: MPCCObservationConfig):
        """
        Initialize observation generator.
        
        Args:
            config: Configuration for observation generation
        """
        self.config = config
        
        # Default normalization bounds
        if config.normalize_bounds is None:
            self.normalize_bounds = [
                [-0.2, 0.2],    # lateral deviation
                [-2.0, 2.0],    # velocity error
                [-5.0, 5.0],    # angular velocity
                [0.0, 1.0],     # curvature (for future points)
                [0.0, 3.0],     # reference velocity (for future points)
            ]
        else:
            self.normalize_bounds = config.normalize_bounds
        
        # Calculate observation dimension
        self.n_observations = self._calculate_observation_dimension()
    
    def _calculate_observation_dimension(self) -> int:
        """Calculate total observation dimension."""
        # Basic state: lateral deviation, velocity error, angular velocity
        basic_dim = 3
        
        # Future trajectory: curvature + reference velocity for each point
        future_dim = self.config.n_anticipation_points * 2
        
        return basic_dim + future_dim
    
    def generate_observation(
        self,
        lateral_deviation: float,
        velocity_error: float,
        angular_velocity: float,
        current_theta: float,
        track_lu_table: np.ndarray,
        track_smax: float
    ) -> np.ndarray:
        """
        Generate observation vector.
        
        Args:
            lateral_deviation: Current lateral deviation from track centerline
            velocity_error: Current velocity error from reference
            angular_velocity: Current angular velocity
            current_theta: Current progress along track
            track_lu_table: Track lookup table
            track_smax: Maximum track progress value
            
        Returns:
            Normalized observation vector
        """
        observation = np.zeros(self.n_observations)
        
        # Basic state observations
        observation[0] = self._normalize_value(lateral_deviation, self.normalize_bounds[0])
        observation[1] = self._normalize_value(velocity_error, self.normalize_bounds[1])
        observation[2] = self._normalize_value(angular_velocity, self.normalize_bounds[2])
        
        # Future trajectory observations
        future_observations = self._get_future_trajectory_observations(
            current_theta, track_lu_table, track_smax
        )
        
        # Add future observations to the observation vector
        start_idx = 3
        observation[start_idx:start_idx + len(future_observations)] = future_observations
        
        return observation
    
    def _normalize_value(self, value: float, bounds: List[float]) -> float:
        """
        Normalize value to [0, 1] range.
        
        Args:
            value: Value to normalize
            bounds: [min, max] bounds
            
        Returns:
            Normalized value in [0, 1]
        """
        min_val, max_val = bounds
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    def _get_future_trajectory_observations(
        self,
        current_theta: float,
        track_lu_table: np.ndarray,
        track_smax: float
    ) -> np.ndarray:
        """
        Get observations about future trajectory.
        
        Args:
            current_theta: Current progress along track
            track_lu_table: Track lookup table
            track_smax: Maximum track progress value
            
        Returns:
            Future trajectory observations
        """
        future_obs = np.zeros(self.config.n_anticipation_points * 2)
        
        # Track variable indices
        trackvars = [
            "sval", "tval", "xtrack", "ytrack", "phitrack",
            "cos(phi)", "sin(phi)", "g_upper", "g_lower",
        ]
        
        # Calculate future theta values
        theta_step = self.config.anticipation_horizon / self.config.n_anticipation_points
        
        for i in range(self.config.n_anticipation_points):
            # Future theta (with wraparound)
            future_theta = current_theta + (i + 1) * theta_step
            if future_theta > track_smax:
                future_theta -= track_smax
            
            # Get track point index
            track_idx = int(future_theta * 100)  # 100 points per meter
            track_idx = max(0, min(track_idx, len(track_lu_table) - 1))
            
            # Calculate curvature at this point
            curvature = self._calculate_local_curvature(track_lu_table, track_idx)
            
            # Reference velocity (could be based on curvature)
            ref_velocity = self._calculate_reference_velocity(curvature)
            
            # Normalize and store
            future_obs[i * 2] = self._normalize_value(curvature, self.normalize_bounds[3])
            future_obs[i * 2 + 1] = self._normalize_value(ref_velocity, self.normalize_bounds[4])
        
        return future_obs
    
    def _calculate_local_curvature(self, track_lu_table: np.ndarray, idx: int) -> float:
        """
        Calculate local curvature at given track index.
        
        Args:
            track_lu_table: Track lookup table
            idx: Track index
            
        Returns:
            Local curvature
        """
        # Track variable indices
        trackvars = [
            "sval", "tval", "xtrack", "ytrack", "phitrack",
            "cos(phi)", "sin(phi)", "g_upper", "g_lower",
        ]
        
        # Get neighboring points for curvature calculation
        n_points = len(track_lu_table)
        
        # Use a small window around the current point
        window = 5
        start_idx = max(0, idx - window)
        end_idx = min(n_points, idx + window + 1)
        
        if end_idx - start_idx < 3:
            return 0.0  # Not enough points for curvature calculation
        
        # Extract x, y coordinates
        x = track_lu_table[start_idx:end_idx, trackvars.index("xtrack")]
        y = track_lu_table[start_idx:end_idx, trackvars.index("ytrack")]
        
        # Calculate curvature using finite differences
        if len(x) >= 3:
            # First and second derivatives
            dx = np.gradient(x)
            dy = np.gradient(y)
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # Curvature formula: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            center_idx = len(dx) // 2
            
            numerator = abs(dx[center_idx] * d2y[center_idx] - dy[center_idx] * d2x[center_idx])
            denominator = (dx[center_idx]**2 + dy[center_idx]**2)**(3/2)
            
            if denominator > 1e-8:
                curvature = numerator / denominator
            else:
                curvature = 0.0
        else:
            curvature = 0.0
        
        return curvature
    
    def _calculate_reference_velocity(self, curvature: float) -> float:
        """
        Calculate reference velocity based on curvature.
        
        Args:
            curvature: Local track curvature
            
        Returns:
            Reference velocity
        """
        # Simple velocity profile: slower on high curvature
        max_velocity = 2.5
        min_velocity = 1.0
        
        # Exponential decay with curvature
        if curvature > 0:
            velocity = min_velocity + (max_velocity - min_velocity) * np.exp(-10 * curvature)
        else:
            velocity = max_velocity
        
        return np.clip(velocity, min_velocity, max_velocity)
    
    def get_observation_info(self) -> dict:
        """
        Get information about the observation space.
        
        Returns:
            Dictionary with observation space information
        """
        return {
            "dimension": self.n_observations,
            "features": [
                "lateral_deviation",
                "velocity_error", 
                "angular_velocity"
            ] + [
                f"future_curvature_{i}" for i in range(self.config.n_anticipation_points)
            ] + [
                f"future_ref_velocity_{i}" for i in range(self.config.n_anticipation_points)
            ],
            "normalize_bounds": self.normalize_bounds,
            "anticipation_points": self.config.n_anticipation_points,
            "anticipation_horizon": self.config.anticipation_horizon
        }


def create_observation_generator(config_dict: dict) -> MPCCObservationGenerator:
    """
    Create observation generator from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        MPCCObservationGenerator instance
    """
    config = MPCCObservationConfig(
        anticipation_horizon=config_dict.get('obs_anticipation_horizon', 20),
        n_anticipation_points=config_dict.get('obs_n_anticipation_points', 5),
        normalize_bounds=config_dict.get('obs_normalize_bounds', None)
    )
    
    return MPCCObservationGenerator(config)