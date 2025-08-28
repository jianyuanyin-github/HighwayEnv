import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import highway_env  # noqa: F401


TRAIN = False
CONTINUE_TRAINING = (
    False  # True: continue training from existing model, False: train from scratch
)
MODEL_TO_LOAD = "model"  # "latest": load newest model, "model": load model.zip, or specific filename like "model_1234567890"

if __name__ == "__main__":
    # GPU optimization configuration
    import torch
    import signal
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Global variable to store model for signal handler
    current_model = None

    def signal_handler(signum, frame):
        print(f"\n\n⚠️  Received signal {signum} (Ctrl+C)")
        if current_model is not None:
            print(" Saving current model before exit...")
            import time

            timestamp = int(time.time())
            emergency_save_path = f"lane_keeping_ppo/model_emergency_{timestamp}"
            current_model.save(emergency_save_path)
            print(f" Model saved as: {emergency_save_path}.zip")
        else:
            print(" No model to save")
        print("Exiting...")
        sys.exit(0)

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    n_envs = 8  # Number of parallel environments
    batch_size = 256  # Increase batch size to utilize GPU better
    env = make_vec_env("lane-keeping-v0", n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Decide whether to continue training based on configuration
    if CONTINUE_TRAINING:
        try:
            model = PPO.load(
                "lane_keeping_ppo/model",
                env=env,
                tensorboard_log="lane_keeping_ppo/",
            )
            print(" Loaded existing model, continuing training...")
        except Exception as e:
            print(f"Cannot load existing model: {e}")
            print("Creating new model to start training...")
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_envs,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                device=device,  # Use GPU
                tensorboard_log="lane_keeping_ppo/",
            )
    else:
        print("✓ Creating new model, training from scratch...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_envs,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.9,
            verbose=2,
            device=device,  # Use GPU
            tensorboard_log="lane_keeping_ppo/",
        )

    # Train the model
    if TRAIN:
        # Store model reference for signal handler (before training starts)
        current_model = model
        print(
            "Signal handlers registered - Press Ctrl+C to save and exit during training"
        )

        # Save checkpoint every 100,000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./lane_keeping_ppo/checkpoints/",
            name_prefix="rl_model",
        )

        model.learn(total_timesteps=int(1e6), callback=checkpoint_callback)

        # Save model with timestamp to avoid overwriting
        import time

        timestamp = int(time.time())
        model_name = f"lane_keeping_ppo/model_{timestamp}"
        model.save(model_name)
        print(f"Model saved as: {model_name}.zip")
        del model

    # Run the algorithm - load model based on configuration
    import glob

    if MODEL_TO_LOAD == "latest":
        # Load the newest timestamped model
        model_files = glob.glob("lane_keeping_ppo/model_*.zip")
        if model_files:
            latest_model = max(model_files)  # Get the newest model file
            model_path = latest_model
            print(f"Loading latest model: {model_path}")
        else:
            print("No timestamped models found, trying model.zip...")
            model_path = "lane_keeping_ppo/model.zip"
    elif MODEL_TO_LOAD == "model":
        # Load default model.zip
        model_path = "lane_keeping_ppo/model.zip"
        print(f"Loading default model: {model_path}")
    else:
        # Load specified model file
        if not MODEL_TO_LOAD.endswith(".zip"):
            model_path = f"lane_keeping_ppo/{MODEL_TO_LOAD}.zip"
        else:
            model_path = f"lane_keeping_ppo/{MODEL_TO_LOAD}"
        print(f"Loading specified model: {model_path}")

    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Available models:")
        all_models = glob.glob("lane_keeping_ppo/model*.zip")
        for m in all_models:
            print(f"  {m}")
        exit(1)

    env = gym.make("lane-keeping-v0", render_mode="rgb_array")
    # env = RecordVideo(
    #     env, video_folder="lane_keeping_ppo/videos", episode_trigger=lambda e: True
    # )
    # env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()