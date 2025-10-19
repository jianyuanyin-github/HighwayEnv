import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MPCCRewardConfig:
    """Configuration for MPCC reward generator."""
    # Gaussian reward function parameters
    lat_dev_sigma: float = 0.1      # Standard deviation for lateral deviation reward
    vel_error_sigma: float = 0.3    # Standard deviation for velocity error reward
    angular_vel_sigma: float = 2.0  # Standard deviation for angular velocity reward
    
    # Reward weights
    lat_dev_weight: float = 0.5     # Weight for lateral deviation
    vel_error_weight: float = 0.3   # Weight for velocity error
    angular_vel_weight: float = 0.1 # Weight for angular velocity
    progress_weight: float = 0.1    # Weight for progress reward
    
    # Penalty parameters
    crash_penalty: float = -1.0     # Penalty for crashing
    boundary_penalty: float = -0.5  # Penalty for approaching boundaries
    
    # Normalization bounds
    lat_dev_bounds: List[float] = None     # [min, max] for lateral deviation
    vel_error_bounds: List[float] = None   # [min, max] for velocity error
    angular_vel_bounds: List[float] = None # [min, max] for angular velocity


class MPCCRewardGenerator:
    """
    Generate rewards for MPCC RL environment.
    
    The reward function encourages:
    - Low lateral deviation from track centerline
    - Low velocity error from reference
    - Smooth angular velocity
    - Forward progress along track
    - Penalizes crashes and boundary violations
    """
    
    def __init__(self, config: MPCCRewardConfig):
        """
        Initialize reward generator.
        
        Args:
            config: Configuration for reward generation
        """
        self.config = config
        
        # Set default bounds if not provided
        if config.lat_dev_bounds is None:
            self.config.lat_dev_bounds = [-0.2, 0.2]
        if config.vel_error_bounds is None:
            self.config.vel_error_bounds = [-2.0, 2.0]
        if config.angular_vel_bounds is None:
            self.config.angular_vel_bounds = [-5.0, 5.0]
    
    def calculate_reward(
        self,
        lateral_deviation: float,
        velocity_error: float,
        angular_velocity: float,
        progress_delta: float,
        crashed: bool = False,
        near_boundary: bool = False,
        step_length: int = 1
    ) -> float:
        """
        Calculate reward based on current performance.
        
        Args:
            lateral_deviation: Current lateral deviation from centerline
            velocity_error: Current velocity error from reference
            angular_velocity: Current angular velocity
            progress_delta: Progress made along track in this step
            crashed: Whether the vehicle crashed
            near_boundary: Whether the vehicle is near track boundaries
            step_length: Number of simulation steps taken
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Crash penalty (immediate and severe)
        if crashed:
            return self.config.crash_penalty
        
        # Boundary penalty
        if near_boundary:
            reward += self.config.boundary_penalty
        
        # Lateral deviation reward (Gaussian)
        lat_dev_reward = self._gaussian_reward(
            lateral_deviation, 
            sigma=self.config.lat_dev_sigma
        )
        reward += self.config.lat_dev_weight * lat_dev_reward
        
        # Velocity error reward (Gaussian)
        vel_error_reward = self._gaussian_reward(
            velocity_error,
            sigma=self.config.vel_error_sigma
        )
        reward += self.config.vel_error_weight * vel_error_reward
        
        # Angular velocity reward (encourage smooth driving)
        angular_vel_reward = self._gaussian_reward(
            angular_velocity,
            sigma=self.config.angular_vel_sigma
        )
        reward += self.config.angular_vel_weight * angular_vel_reward
        
        # Progress reward (encourage forward movement)
        progress_reward = max(0.0, progress_delta)  # Only positive progress
        reward += self.config.progress_weight * progress_reward
        
        # Scale reward by step length
        reward = reward * step_length
        
        return reward
    
    def calculate_episode_reward(
        self,
        lateral_deviations: List[float],
        velocity_errors: List[float],
        angular_velocities: List[float],
        progress_values: List[float],
        crash_occurred: bool = False
    ) -> Dict[str, float]:
        """
        Calculate episode-level reward metrics.
        
        Args:
            lateral_deviations: List of lateral deviations throughout episode
            velocity_errors: List of velocity errors throughout episode
            angular_velocities: List of angular velocities throughout episode
            progress_values: List of progress values throughout episode
            crash_occurred: Whether a crash occurred during the episode
            
        Returns:
            Dictionary with episode reward metrics
        """
        if len(lateral_deviations) == 0:
            return {
                "total_reward": 0.0,
                "avg_lat_dev": 0.0,
                "avg_vel_error": 0.0,
                "avg_angular_vel": 0.0,
                "total_progress": 0.0,
                "crashed": crash_occurred
            }
        
        # Calculate RMS values
        rms_lat_dev = self._calculate_rms(lateral_deviations)
        rms_vel_error = self._calculate_rms(velocity_errors)
        rms_angular_vel = self._calculate_rms(angular_velocities)
        
        # Calculate total progress
        total_progress = progress_values[-1] - progress_values[0] if len(progress_values) > 1 else 0.0
        
        # Calculate episode reward
        episode_reward = 0.0
        
        # Base rewards using RMS values
        lat_dev_reward = self._gaussian_reward(rms_lat_dev, self.config.lat_dev_sigma)
        vel_error_reward = self._gaussian_reward(rms_vel_error, self.config.vel_error_sigma)
        angular_vel_reward = self._gaussian_reward(rms_angular_vel, self.config.angular_vel_sigma)
        
        episode_reward += self.config.lat_dev_weight * lat_dev_reward
        episode_reward += self.config.vel_error_weight * vel_error_reward
        episode_reward += self.config.angular_vel_weight * angular_vel_reward
        episode_reward += self.config.progress_weight * total_progress
        
        # Apply crash penalty
        if crash_occurred:
            episode_reward += self.config.crash_penalty
        
        return {
            "total_reward": episode_reward,
            "lat_dev_reward": lat_dev_reward,
            "vel_error_reward": vel_error_reward,
            "angular_vel_reward": angular_vel_reward,
            "progress_reward": total_progress,
            "rms_lat_dev": rms_lat_dev,
            "rms_vel_error": rms_vel_error,
            "rms_angular_vel": rms_angular_vel,
            "total_progress": total_progress,
            "crashed": crash_occurred
        }
    
    def _gaussian_reward(self, value: float, sigma: float, center: float = 0.0) -> float:
        """
        Calculate Gaussian reward function.
        
        Args:
            value: Input value
            sigma: Standard deviation of Gaussian
            center: Center of Gaussian (target value)
            
        Returns:
            Reward value in [0, 1]
        """
        return np.exp(-0.5 * ((value - center) / sigma) ** 2)
    
    def _calculate_rms(self, values: List[float]) -> float:
        """
        Calculate root mean square of values.
        
        Args:
            values: List of values
            
        Returns:
            RMS value
        """
        if len(values) == 0:
            return 0.0
        
        return np.sqrt(np.mean(np.array(values) ** 2))
    
    def get_reward_info(self) -> Dict[str, Any]:
        """
        Get information about the reward function.
        
        Returns:
            Dictionary with reward function information
        """
        return {
            "components": {
                "lateral_deviation": {
                    "weight": self.config.lat_dev_weight,
                    "sigma": self.config.lat_dev_sigma,
                    "bounds": self.config.lat_dev_bounds
                },
                "velocity_error": {
                    "weight": self.config.vel_error_weight,
                    "sigma": self.config.vel_error_sigma,
                    "bounds": self.config.vel_error_bounds
                },
                "angular_velocity": {
                    "weight": self.config.angular_vel_weight,
                    "sigma": self.config.angular_vel_sigma,
                    "bounds": self.config.angular_vel_bounds
                },
                "progress": {
                    "weight": self.config.progress_weight
                }
            },
            "penalties": {
                "crash": self.config.crash_penalty,
                "boundary": self.config.boundary_penalty
            }
        }


class AdaptiveRewardGenerator(MPCCRewardGenerator):
    """
    Adaptive reward generator that adjusts reward weights based on performance.
    """
    
    def __init__(self, config: MPCCRewardConfig, adaptation_rate: float = 0.01):
        """
        Initialize adaptive reward generator.
        
        Args:
            config: Base reward configuration
            adaptation_rate: Rate of adaptation for reward weights
        """
        super().__init__(config)
        self.adaptation_rate = adaptation_rate
        self.performance_history = {
            "lateral_deviation": [],
            "velocity_error": [],
            "angular_velocity": []
        }
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """
        Update reward weights based on recent performance.
        
        Args:
            performance_metrics: Dictionary with performance metrics
        """
        # Update performance history
        if "rms_lat_dev" in performance_metrics:
            self.performance_history["lateral_deviation"].append(performance_metrics["rms_lat_dev"])
        if "rms_vel_error" in performance_metrics:
            self.performance_history["velocity_error"].append(performance_metrics["rms_vel_error"])
        if "rms_angular_vel" in performance_metrics:
            self.performance_history["angular_velocity"].append(performance_metrics["rms_angular_vel"])
        
        # Keep only recent history
        max_history = 100
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
        
        # Adapt weights based on relative performance
        if len(self.performance_history["lateral_deviation"]) > 10:
            # Increase weight for components with poor performance
            recent_lat_dev = np.mean(self.performance_history["lateral_deviation"][-10:])
            recent_vel_error = np.mean(self.performance_history["velocity_error"][-10:])
            
            # Simple adaptation: increase weight if performance is poor
            if recent_lat_dev > 0.1:  # Poor lateral tracking
                self.config.lat_dev_weight += self.adaptation_rate
            if recent_vel_error > 0.5:  # Poor velocity tracking
                self.config.vel_error_weight += self.adaptation_rate
            
            # Normalize weights
            total_weight = (self.config.lat_dev_weight + 
                          self.config.vel_error_weight + 
                          self.config.angular_vel_weight +
                          self.config.progress_weight)
            
            if total_weight > 1.0:
                self.config.lat_dev_weight /= total_weight
                self.config.vel_error_weight /= total_weight
                self.config.angular_vel_weight /= total_weight
                self.config.progress_weight /= total_weight


def create_reward_generator(config_dict: Dict[str, Any]) -> MPCCRewardGenerator:
    """
    Create reward generator from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        MPCCRewardGenerator instance
    """
    config = MPCCRewardConfig(
        lat_dev_sigma=config_dict.get('rew_lat_dev_sigma', 0.1),
        vel_error_sigma=config_dict.get('rew_vel_error_sigma', 0.3),
        angular_vel_sigma=config_dict.get('rew_angular_vel_sigma', 2.0),
        lat_dev_weight=config_dict.get('rew_lat_dev_weight', 0.5),
        vel_error_weight=config_dict.get('rew_vel_error_weight', 0.3),
        angular_vel_weight=config_dict.get('rew_angular_vel_weight', 0.1),
        progress_weight=config_dict.get('rew_progress_weight', 0.1),
        crash_penalty=config_dict.get('rew_crash_penalty', -1.0),
        boundary_penalty=config_dict.get('rew_boundary_penalty', -0.5),
        lat_dev_bounds=config_dict.get('rew_lims_lat_dev', None),
        vel_error_bounds=config_dict.get('rew_lims_vel_dev', None),
        angular_vel_bounds=config_dict.get('rew_lims_angular_vel', None)
    )
    
    # Check if adaptive reward is requested
    if config_dict.get('use_adaptive_reward', False):
        adaptation_rate = config_dict.get('adaptive_reward_rate', 0.01)
        return AdaptiveRewardGenerator(config, adaptation_rate)
    else:
        return MPCCRewardGenerator(config)