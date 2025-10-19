#!/usr/bin/env python3
"""
Training script for MPCC reinforcement learning.
"""

import os
import sys
import numpy as np
import torch
import yaml
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# MPCC RL components
from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.mpcc_environment import MPCCEnvironment
from Learning_To_Adapt.SafeRL_WMPC.helpers import load_config


class MPCCRLTrainer:
    """Trainer for MPCC reinforcement learning."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.model = None
        self.train_env = None
        self.eval_env = None
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        print("MPCC RL Trainer initialized")
        print(f"Config: {config_path}")
        print(f"Training steps: {self.config.get('n_training_steps', 100000)}")
    
    def setup_directories(self):
        """Setup training directories."""
        self.model_dir = Path(self.config.get('model_dir', 'Learning_To_Adapt/SafeRL_WMPC/_models'))
        self.log_dir = Path(self.config.get('log_dir', 'Learning_To_Adapt/SafeRL_WMPC/_logs'))
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Model directory: {self.model_dir}")
        print(f"Log directory: {self.log_dir}")
    
    def setup_logging(self):
        """Setup logging for training."""
        self.tensorboard_log = str(self.log_dir / "tensorboard")
        print(f"Tensorboard logs: {self.tensorboard_log}")
    
    def create_env(self, env_id: int = 0, evaluation: bool = False) -> MPCCEnvironment:
        """
        Create MPCC environment.
        
        Args:
            env_id: Environment ID
            evaluation: Whether this is an evaluation environment
            
        Returns:
            MPCC environment
        """
        # Select trajectory
        trajectories = self.config.get('trajectories', ['slider'])
        trajectory = trajectories[env_id % len(trajectories)]
        
        # Environment configuration
        env_config = {
            'n_mpc_steps': self.config.get('n_mpc_steps', 10),
            'max_lat_dev': self.config.get('max_lat_dev', 0.2),
            'episode_length': self.config.get('episode_length', 100),
            'target_velocity': self.config.get('target_velocity', 1.5),
            'obs_anticipation_horizon': self.config.get('obs_anticipation_horizon', 20),
            'obs_n_anticipation_points': self.config.get('obs_n_anticipation_points', 5),
            'actions_file': self.config.get('actions_file', None)
        }
        
        return MPCCEnvironment(
            config=env_config,
            trajectory=trajectory,
            random_restarts=not evaluation,
            evaluation_env=evaluation
        )
    
    def create_training_env(self):
        """Create training environment."""
        n_envs = self.config.get('n_environments', 4)
        
        # Create multiple environments
        def make_env(env_id: int):
            def _init():
                return self.create_env(env_id, evaluation=False)
            return _init
        
        # Use SubprocVecEnv for parallel environments
        if n_envs > 1:
            self.train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        else:
            self.train_env = DummyVecEnv([make_env(0)])
        
        # Add monitoring
        self.train_env = VecMonitor(
            self.train_env,
            filename=str(self.log_dir / "training_monitor.csv")
        )
        
        print(f"Created training environment with {n_envs} parallel environments")
    
    def create_evaluation_env(self):
        """Create evaluation environment."""
        n_eval_envs = self.config.get('n_envs_eval', 1)
        
        def make_eval_env(env_id: int):
            def _init():
                return self.create_env(env_id, evaluation=True)
            return _init
        
        if n_eval_envs > 1:
            self.eval_env = SubprocVecEnv([make_eval_env(i) for i in range(n_eval_envs)])
        else:
            self.eval_env = DummyVecEnv([make_eval_env(0)])
        
        # Add monitoring
        self.eval_env = VecMonitor(
            self.eval_env,
            filename=str(self.log_dir / "evaluation_monitor.csv")
        )
        
        print(f"Created evaluation environment with {n_eval_envs} parallel environments")
    
    def create_model(self):
        """Create PPO model."""
        # PPO hyperparameters
        learning_rate = self.config.get('learning_rate', 3e-4)
        n_steps = self.config.get('n_steps', 256)
        batch_size = self.config.get('batch_size', 2048)
        n_epochs = self.config.get('n_epochs', 5)
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)
        clip_range = self.config.get('clip_range', 0.2)
        ent_coef = self.config.get('ent_coef', 0.01)
        vf_coef = self.config.get('vf_coef', 0.5)
        max_grad_norm = self.config.get('max_grad_norm', 0.5)
        
        # Network architecture
        net_arch = self.config.get('net_arch', [64, 64])
        
        # Policy kwargs
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": torch.nn.ReLU
        }
        
        # Create model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.tensorboard_log,
            verbose=1
        )
        
        print("Created PPO model")
        print(f"Learning rate: {learning_rate}")
        print(f"Network architecture: {net_arch}")
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_freq = self.config.get('checkpoint_freq', 10000)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(self.model_dir),
            name_prefix="mpcc_rl_model"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if self.eval_env is not None:
            eval_freq = self.config.get('evaluation_frequency', 5000)
            n_eval_episodes = self.config.get('n_eval_episodes', 5)
            
            eval_callback = EvalCallback(
                eval_env=self.eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                best_model_save_path=str(self.model_dir / "best_model"),
                log_path=str(self.log_dir / "evaluation"),
                deterministic=True,
                verbose=1
            )
            callbacks.append(eval_callback)
        
        return callbacks
    
    def train(self):
        """Train the model."""
        print("\nStarting training...")
        
        # Create environments
        self.create_training_env()
        self.create_evaluation_env()
        
        # Create model
        self.create_model()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Training parameters
        total_timesteps = self.config.get('n_training_steps', 100000)
        
        print(f"Training for {total_timesteps} timesteps")
        
        # Train model
        start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name="mpcc_rl_training"
            )
            
            training_time = time.time() - start_time
            print(f"\nTraining completed in {training_time:.2f} seconds")
            
            # Save final model
            final_model_path = self.model_dir / "mpcc_rl_final_model"
            self.model.save(str(final_model_path))
            print(f"Final model saved to: {final_model_path}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            
            # Save current model
            interrupted_model_path = self.model_dir / "mpcc_rl_interrupted_model"
            self.model.save(str(interrupted_model_path))
            print(f"Model saved to: {interrupted_model_path}")
        
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise
        
        finally:
            # Close environments
            if self.train_env:
                self.train_env.close()
            if self.eval_env:
                self.eval_env.close()
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
        """
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path)
        print("Model loaded successfully")
    
    def evaluate_model(self, model_path: str = None, n_episodes: int = 10):
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to model file (if None, use current model)
            n_episodes: Number of episodes to evaluate
        """
        if model_path:
            self.load_model(model_path)
        
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        print(f"\nEvaluating model for {n_episodes} episodes...")
        
        # Create evaluation environment
        self.create_evaluation_env()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Length = {episode_length}")
        
        # Summary statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        print(f"\nEvaluation Results:")
        print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"Mean episode length: {mean_length:.1f}")
        
        # Close evaluation environment
        self.eval_env.close()
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }


def main():
    """Main training function."""
    # Change to project root directory
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # Configuration file
    config_path = "Learning_To_Adapt/SafeRL_WMPC/_config/rl_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please ensure the configuration file exists.")
        return
    
    # Create trainer
    trainer = MPCCRLTrainer(config_path)
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()