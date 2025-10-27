#!/usr/bin/env python3
"""Test the trained pursuit model to check speed variations."""

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import numpy as np


def test_trained_model():
    """Test trained model and analyze speed behavior."""

    # Load trained model
    model = PPO.load("racetrack_pursuit_ppo/model_1760955688.zip")

    # Create environment
    env = gym.make("racetrack-single-pursuit-v0", render_mode="human")

    print("=" * 70)
    print("Testing Trained Pure Pursuit Model")
    print("=" * 70)

    # Run 3 episodes
    for episode in range(3):
        obs, info = env.reset()
        done = truncated = False
        step = 0
        total_reward = 0

        # Track speeds by lane type
        straight_speeds = []
        curve_speeds = []
        actions_taken = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }  # SLOWER, IDLE, FASTER, LEFT, RIGHT

        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}")
        print(f"{'='*70}")

        while not (done or truncated) and step < 1000:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            actions_taken[int(action)] += 1

            # Step
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            total_reward += reward

            vehicle = env.unwrapped.vehicle
            lane = vehicle.lane

            # Classify lane type
            from highway_env.road.lane import CircularLane

            if isinstance(lane, CircularLane):
                curve_speeds.append(vehicle.speed)
                lane_type = f"Curve(r={lane.radius:.0f})"
            else:
                straight_speeds.append(vehicle.speed)
                lane_type = "Straight"

            # Print some steps
            if step % 100 == 0:
                action_names = {
                    0: "SLOWER",
                    1: "IDLE",
                    2: "FASTER",
                    3: "LEFT",
                    4: "RIGHT",
                }
                lane_idx = vehicle.lane_index
                print(
                    f"  Step {step:4d} | {lane_idx[0]}→{lane_idx[1]} | "
                    f"{lane_type:15s} | Speed: {vehicle.speed:5.2f} m/s | "
                    f"Target: {vehicle.target_speed:5.2f} m/s | "
                    f"Action: {action_names[int(action)]}"
                )

            env.render()

        print(f"\n{'='*70}")
        print(f"Episode {episode + 1} Summary")
        print(f"{'='*70}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {step}")

        if straight_speeds:
            print(f"\n  Straight sections:")
            print(f"    Min speed:  {min(straight_speeds):5.2f} m/s")
            print(f"    Max speed:  {max(straight_speeds):5.2f} m/s")
            print(f"    Avg speed:  {np.mean(straight_speeds):5.2f} m/s")
            print(f"    Std dev:    {np.std(straight_speeds):5.2f} m/s")

        if curve_speeds:
            print(f"\n  Curve sections:")
            print(f"    Min speed:  {min(curve_speeds):5.2f} m/s")
            print(f"    Max speed:  {max(curve_speeds):5.2f} m/s")
            print(f"    Avg speed:  {np.mean(curve_speeds):5.2f} m/s")
            print(f"    Std dev:    {np.std(curve_speeds):5.2f} m/s")

        print(f"\n  Actions taken:")
        action_names = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LEFT", 4: "RIGHT"}
        for action_id, count in actions_taken.items():
            pct = 100 * count / step
            print(f"    {action_names[action_id]:8s}: {count:4d} times ({pct:5.1f}%)")

        # Speed variation check
        if straight_speeds and curve_speeds:
            avg_straight = np.mean(straight_speeds)
            avg_curve = np.mean(curve_speeds)
            speed_diff = avg_straight - avg_curve
            print(f"\n  ✓ Speed difference (Straight - Curve): {speed_diff:+.2f} m/s")

            if abs(speed_diff) > 1.0:
                print(f"  ✅ Agent learned to adjust speed! (difference > 1 m/s)")
            else:
                print(f"  ⚠️  Agent uses mostly constant speed (difference < 1 m/s)")

    env.close()
    print(f"\n{'='*70}")
    print("Test completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_trained_model()
