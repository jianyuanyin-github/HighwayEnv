import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import highway_env  # noqa: F401


TRAIN = True
CONTINUE_TRAINING = (
    False  # True: continue training from existing model, False: train from scratch
)

if __name__ == "__main__":
    # GPU optimization configuration
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_envs = 8  # Number of parallel environments
    batch_size = 256  # Increase batch size to utilize GPU better
    env = make_vec_env("racetrack-v0", n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Decide whether to continue training based on configuration
    if CONTINUE_TRAINING:
        try:
            model = PPO.load(
                "racetrack_ppo/model", env=env, tensorboard_log="racetrack_ppo/"
            )
            print("✓ Loaded existing model, continuing training...")
        except Exception as e:
            print(f"✗ Cannot load existing model: {e}")
            print("✓ Creating new model to start training...")
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
                tensorboard_log="racetrack_ppo/",
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
            tensorboard_log="racetrack_ppo/",
        )
    # Train the model
    if TRAIN:
        # Save checkpoint every 100,000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path="./racetrack_ppo/checkpoints/",
            name_prefix="rl_model",
        )

        model.learn(total_timesteps=int(1e5), callback=checkpoint_callback)
        model.save("racetrack_ppo/model")
        del model

    # Run the algorithm
    model = PPO.load("racetrack_ppo/model", env=env)

    env = gym.make("racetrack-v0", render_mode="rgb_array")
    # env = RecordVideo(
    #     env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True
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
