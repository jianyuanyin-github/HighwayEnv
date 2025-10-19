import gymnasium as gym
import torch
import numpy as np
import random
import yaml
import os
import sys

from gymnasium import spaces
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from MPCC.MPCC_class import MPCC
import tracks.InterpolateTrack as InterpolateTrack
from simulator.python_sim_utils import plotter, plot_pajecka, compute_objective


@dataclass
class MPCCSimulationState:
    """State information for MPCC simulation."""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    orientation: float    # phi
    angular_velocity: float  # omega
    lateral_deviation: float
    velocity_error: float
    theta: float  # progress along track
    

class MPCCEnvironment(gym.Env):
    """
    Reinforcement Learning Environment for MPCC parameter tuning.
    
    This environment allows an RL agent to select MPCC parameters
    and observes the resulting performance.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        config: Dict[str, Any],
        trajectory: str = "slider",
        random_restarts: bool = True,
        evaluation_env: bool = False
    ):
        """
        Initialize MPCC RL environment.
        
        Args:
            config: Configuration dictionary
            trajectory: Name of trajectory to use
            random_restarts: Whether to use random start positions
            evaluation_env: Whether this is an evaluation environment
        """
        super().__init__()
        
        # Environment settings
        self.config = config
        self.trajectory = trajectory
        self.random_restarts = random_restarts
        self.evaluation_env = evaluation_env
        
        # Simulation parameters
        self.n_mpc_steps = config.get('n_mpc_steps', 10)
        self.max_lat_dev = config.get('max_lat_dev', 0.2)
        self.episode_length = config.get('episode_length', 100)
        self.target_velocity = config.get('target_velocity', 1.5)
        
        # Load vehicle and MPCC parameters
        self._load_parameters()
        
        # Setup track
        self._setup_track()
        
        # Setup action space (discrete parameter sets)
        self._setup_action_space()
        
        # Setup observation space
        self._setup_observation_space()
        
        # Initialize state
        self.episode_steps = 0
        self.crash_counter = 0
        self.current_state = None
        self.mpcc_controller = None
        
        # Performance tracking
        self.lateral_deviations = []
        self.velocity_errors = []
        self.rewards = []
        
        # Reset to initial state
        self.reset()
    
    def _load_parameters(self):
        """Load vehicle and MPCC parameters from config files."""
        try:
            with open("config/vehicleparams.yaml") as file:
                self.vehicle_params = yaml.load(file, Loader=yaml.FullLoader)
            
            with open("config/mpccparams.yaml") as file:
                self.mpcc_params = yaml.load(file, Loader=yaml.FullLoader)
                
        except FileNotFoundError as e:
            print(f"Warning: Could not load parameter files: {e}")
            # Use default parameters
            self.vehicle_params = {
                "lf": 0.03, "lr": 0.03, "m": 0.06, "Iz": 1e-5,
                "B": 4.0, "C": 1.0, "D": 1.0, "mu": 0.8
            }
            self.mpcc_params = {
                "N": 35, "Tsim": 40, "Tf": 1.0,
                "Qc": 1000.0, "Ql": 1500.0, "Q_theta": 10.0,
                "R_d": 0.01, "R_delta": 0.01
            }
    
    def _setup_track(self):
        """Setup track lookup table."""
        try:
            track_file = f"tracks/{self.trajectory}"
            track_lu_table, smax = InterpolateTrack.generatelookuptable(track_file)
            
            self.track = {
                "track_lu_table": track_lu_table,
                "smax": smax,
                "r": 0.2,  # track width
            }
            
            self.track_length = len(track_lu_table)
            
        except Exception as e:
            print(f"Warning: Could not load track {self.trajectory}: {e}")
            # Create dummy track
            self.track = {
                "track_lu_table": np.zeros((1000, 9)),
                "smax": 100.0,
                "r": 0.2,
            }
            self.track_length = 1000
    
    def _setup_action_space(self):
        """Setup discrete action space for parameter selection."""
        # Define parameter sets to choose from
        # Each parameter set is [Qc, Ql, Q_theta, R_d, R_delta]
        
        if 'actions_file' in self.config and self.config['actions_file'] is not None:
            # Load from file if specified
            try:
                with open(self.config['actions_file'], 'r') as file:
                    lines = file.readlines()
                    
                parameter_sets = []
                for line in lines:
                    params = [float(x.strip()) for x in line.strip().split(',')]
                    parameter_sets.append(params)
                    
                self.parameter_sets = torch.tensor(parameter_sets)
                
            except FileNotFoundError:
                print(f"Warning: Actions file not found: {self.config['actions_file']}")
                self.parameter_sets = self._create_default_parameter_sets()
        else:
            # Create default parameter sets
            self.parameter_sets = self._create_default_parameter_sets()
        
        self.n_actions = len(self.parameter_sets)
        self.action_space = spaces.Discrete(self.n_actions)
    
    def _create_default_parameter_sets(self) -> torch.Tensor:
        """Create default parameter sets for action space."""
        # Create a grid of parameter values
        Qc_values = [500, 1000, 1500, 2000]
        Ql_values = [1000, 1500, 2000, 2500]
        Q_theta_values = [5, 10, 20, 30]
        R_d_values = [0.005, 0.01, 0.02, 0.05]
        R_delta_values = [0.005, 0.01, 0.02, 0.05]
        
        parameter_sets = []
        
        # Create combinations (not all combinations to keep action space reasonable)
        for qc in Qc_values:
            for ql in Ql_values:
                for q_theta in Q_theta_values[:2]:  # Use only first 2 values
                    for r_d in R_d_values[:2]:  # Use only first 2 values
                        for r_delta in R_delta_values[:2]:  # Use only first 2 values
                            parameter_sets.append([qc, ql, q_theta, r_d, r_delta])
        
        return torch.tensor(parameter_sets)
    
    def _setup_observation_space(self):
        """Setup observation space."""
        # Observation includes:
        # - Current lateral deviation (normalized)
        # - Current velocity error (normalized)
        # - Current progress along track (normalized)
        # - Future reference trajectory information
        
        obs_dim = 3  # Basic state: lat_dev, vel_error, progress
        
        # Add future reference trajectory points
        n_future_points = self.config.get('obs_n_anticipation_points', 5)
        horizon = self.config.get('obs_anticipation_horizon', 20)
        
        # Each future point contributes: curvature, reference_velocity
        obs_dim += n_future_points * 2
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.n_future_points = n_future_points
        self.horizon = horizon
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (parameter set index)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Update episode steps
        self.episode_steps += 1
        
        # Select parameter set
        params = self.parameter_sets[action]
        
        # Update MPCC controller with new parameters
        self._update_mpcc_parameters(params)
        
        # Run MPCC steps
        crashed = False
        if self.mpcc_controller is not None:
            for i in range(self.n_mpc_steps):
                try:
                    # Get obstacle info (no obstacle for now)
                    obstacle_info = self._get_default_obstacle_info()
                    
                    # Update MPCC controller
                    z_current = self.mpcc_controller.update(obstacle_info)
                    
                    # Extract current state
                    current_state = self._extract_state(z_current)
                    
                    # Check crash condition
                    if self._check_crash_condition(current_state):
                        crashed = True
                        self.crash_counter += 1
                        break
                    
                    # Update performance tracking
                    self.lateral_deviations.append(abs(current_state.lateral_deviation))
                    self.velocity_errors.append(abs(current_state.velocity_error))
                    
                    # Update current state
                    self.current_state = current_state
                    
                except Exception as e:
                    print(f"Error in MPCC step: {e}")
                    crashed = True
                    break
        else:
            # No controller available, simulate dummy behavior
            crashed = True
        
        # Calculate reward
        reward = self._calculate_reward()
        self.rewards.append(reward)
        
        # Get observation
        observation = self._get_observation()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = crashed or (self.episode_steps >= self.episode_length)
        
        # Info dictionary
        info = {
            'lateral_deviation': current_state.lateral_deviation if current_state else 0.0,
            'velocity_error': current_state.velocity_error if current_state else 0.0,
            'crashed': crashed,
            'episode_steps': self.episode_steps
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode
        self.episode_steps = 0
        self.lateral_deviations = []
        self.velocity_errors = []
        self.rewards = []
        
        # Choose starting position
        if self.random_restarts:
            # Choose random start position
            start_positions = [0, 100, 200, 300, 400, 500, 600, 700, 800]
            start_idx = random.choice(start_positions)
            start_idx = min(start_idx, self.track_length - 100)  # Ensure valid index
        else:
            start_idx = 0
        
        # Initialize MPCC controller
        self._initialize_mpcc_controller(start_idx)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial state
        if self.mpcc_controller:
            z_current = self.mpcc_controller.z_current
            self.current_state = self._extract_state(z_current)
        else:
            self.current_state = None
        
        info = {'start_idx': start_idx}
        
        return observation, info
    
    def _initialize_mpcc_controller(self, start_idx: int):
        """Initialize MPCC controller at given start position."""
        try:
            # Create MPCC controller
            self.mpcc_controller = MPCC(
                self.track, 
                self.mpcc_params["Tsim"], 
                self.vehicle_params, 
                self.mpcc_params
            )
            
            # Get initial state
            xinit = self._get_initial_state(start_idx)
            
            # Get obstacle info
            obstacle_info = self._get_default_obstacle_info()
            
            # Initialize trajectory
            z_current = self.mpcc_controller.initialize_trajectory(xinit, obstacle_info, start_idx)
            
        except Exception as e:
            print(f"Error initializing MPCC controller: {e}")
            self.mpcc_controller = None
    
    def _get_initial_state(self, start_idx: int) -> np.ndarray:
        """Get initial state for given start index."""
        # Track variables
        trackvars = [
            "sval", "tval", "xtrack", "ytrack", "phitrack",
            "cos(phi)", "sin(phi)", "g_upper", "g_lower",
        ]
        
        # Ensure start_idx is within bounds
        start_idx = max(0, min(start_idx, self.track_length - 1))
        
        track_lu_table = self.track["track_lu_table"]
        
        xt0 = track_lu_table[start_idx, trackvars.index("xtrack")]
        yt0 = track_lu_table[start_idx, trackvars.index("ytrack")]
        phit0 = track_lu_table[start_idx, trackvars.index("phitrack")]
        theta_hat0 = track_lu_table[start_idx, trackvars.index("sval")]
        
        # Initial velocity
        vx0 = 1.0
        
        return np.array([xt0, yt0, phit0, vx0, 0.0, 0.0, 0.0, 0.0, theta_hat0])
    
    def _get_default_obstacle_info(self) -> Dict[str, Any]:
        """Get default obstacle information (no obstacle)."""
        return {
            "x_ob": 1000.0,  # Far away
            "y_ob": 1000.0,
            "phi_ob": 0.0,
            "l_ob": 0.1,
            "w_ob": 0.1,
        }
    
    def _update_mpcc_parameters(self, params: torch.Tensor):
        """Update MPCC controller parameters."""
        if self.mpcc_controller:
            self.mpcc_controller.Qc = params[0].item()
            self.mpcc_controller.Ql = params[1].item()
            self.mpcc_controller.Q_theta = params[2].item()
            self.mpcc_controller.R_d = params[3].item()
            self.mpcc_controller.R_delta = params[4].item()
    
    def _extract_state(self, z_current: np.ndarray) -> MPCCSimulationState:
        """Extract simulation state from MPCC solution."""
        # Current state is first row
        current_z = z_current[0]
        
        # Extract state components
        position = current_z[3:5]  # [posx, posy]
        velocity = current_z[6:8]  # [vx, vy]
        orientation = current_z[5]  # phi
        angular_velocity = current_z[8]  # omega
        theta = current_z[11]  # progress along track
        
        # Calculate lateral deviation
        lateral_deviation = self._calculate_lateral_deviation(position, theta)
        
        # Calculate velocity error
        velocity_error = np.linalg.norm(velocity) - self.target_velocity
        
        return MPCCSimulationState(
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            lateral_deviation=lateral_deviation,
            velocity_error=velocity_error,
            theta=theta
        )
    
    def _calculate_lateral_deviation(self, position: np.ndarray, theta: float) -> float:
        """Calculate lateral deviation from track centerline."""
        # Simple approximation using theta as index
        track_lu_table = self.track["track_lu_table"]
        
        # Find closest track point
        idx = int(theta * 100)  # 100 points per meter
        idx = max(0, min(idx, len(track_lu_table) - 1))
        
        # Track variables
        trackvars = [
            "sval", "tval", "xtrack", "ytrack", "phitrack",
            "cos(phi)", "sin(phi)", "g_upper", "g_lower",
        ]
        
        track_pos = track_lu_table[idx, 2:4]  # [xtrack, ytrack]
        track_phi = track_lu_table[idx, 4]     # phitrack
        
        # Calculate deviation in track frame
        dx = position[0] - track_pos[0]
        dy = position[1] - track_pos[1]
        
        # Transform to track frame
        lateral_deviation = -dx * np.sin(track_phi) + dy * np.cos(track_phi)
        
        return lateral_deviation
    
    def _check_crash_condition(self, state: MPCCSimulationState) -> bool:
        """Check if vehicle has crashed."""
        return abs(state.lateral_deviation) > self.max_lat_dev
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if completed track (for full lap mode)
        if self.current_state and hasattr(self, 'target_theta'):
            return self.current_state.theta >= self.target_theta
        
        # Otherwise, terminate based on episode length
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on performance."""
        if len(self.lateral_deviations) == 0:
            return -0.1  # Small penalty for no progress
        
        # Recent performance (last few steps)
        recent_steps = min(10, len(self.lateral_deviations))
        recent_lat_dev = np.mean(self.lateral_deviations[-recent_steps:])
        recent_vel_error = np.mean(self.velocity_errors[-recent_steps:])
        
        # Reward based on Gaussian functions
        lat_dev_reward = np.exp(-0.5 * (recent_lat_dev / 0.1)**2)
        vel_error_reward = np.exp(-0.5 * (recent_vel_error / 0.3)**2)
        
        # Combined reward
        reward = 0.7 * lat_dev_reward + 0.3 * vel_error_reward
        
        # Penalty for crash
        if self.current_state and self._check_crash_condition(self.current_state):
            reward -= 1.0
        
        # Ensure reward is finite and reasonable
        reward = np.clip(reward, -2.0, 2.0)
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.observation_space.shape[0])
        
        if self.current_state is None:
            return obs
        
        # Basic state (normalized to [0, 1])
        obs[0] = np.clip((self.current_state.lateral_deviation + self.max_lat_dev) / (2 * self.max_lat_dev), 0, 1)
        obs[1] = np.clip((self.current_state.velocity_error + 2) / 4, 0, 1)  # velocity error ∈ [-2, 2]
        obs[2] = np.clip(self.current_state.theta / self.track["smax"], 0, 1)
        
        # Future reference trajectory (simplified)
        for i in range(self.n_future_points):
            # Future curvature (placeholder)
            obs[3 + i * 2] = 0.5
            # Future reference velocity (placeholder)
            obs[3 + i * 2 + 1] = 0.5
        
        return obs.astype(np.float32)
    
    def render(self, mode: str = "human"):
        """Render the environment (placeholder)."""
        pass
    
    def close(self):
        """Close the environment."""
        pass