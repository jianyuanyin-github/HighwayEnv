#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import highway_env


def test_complex_racetrack():
    """Test the new complex racetrack environment"""

    # Create environment
    env = gym.make("racetrack-single-v0", render_mode="human")

    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(
        f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}"
    )

    # Test episodes
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        done = truncated = False
        step = 0
        total_reward = 0

        while not (done or truncated) and step < 1000:
            # Test both longitudinal and lateral control
            if step < 100:
                # Accelerate forward
                action = np.array([0.5, 0.0])  # [throttle, steering]
            elif step < 200:
                # Turn right while maintaining speed
                action = np.array([0.2, 0.3])  # [throttle, steering_right]
            elif step < 300:
                # Turn left
                action = np.array([0.2, -0.3])  # [throttle, steering_left]
            else:
                # Random actions to test full control
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if step % 50 == 0:
                print(
                    f"Step {step}: reward={reward:.3f}, total_reward={total_reward:.3f}"
                )
                if hasattr(env.unwrapped.vehicle, "speed"):
                    print(f"  Vehicle speed: {env.unwrapped.vehicle.speed:.2f}")
                    print(f"  On road: {env.unwrapped.vehicle.on_road}")

            step += 1

        print(f"Episode {episode + 1} finished after {step} steps")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Terminated: {done}, Truncated: {truncated}")

    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_complex_racetrack()
