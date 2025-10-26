#!/usr/bin/env python3
"""
Visualization script for the complex track.
Shows the track layout and runs a test episode.
"""
import gymnasium as gym
import numpy as np
import highway_env
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

def visualize_track_layout():
    """Visualize the track layout using matplotlib."""
    print("Creating environment and extracting track data...")
    env = gym.make("racetrack-complex-v0", render_mode="rgb_array")
    env.reset()

    road_network = env.unwrapped.road.network

    fig, ax = plt.subplots(figsize=(14, 14))

    # Plot all lanes
    for start_node in road_network.graph.keys():
        for end_node in road_network.graph[start_node].keys():
            lanes = road_network.graph[start_node][end_node]
            for lane in lanes:
                # Sample points along the lane
                if hasattr(lane, 'length'):
                    # Straight lane
                    num_points = max(10, int(lane.length / 2))
                    s_vals = np.linspace(0, lane.length, num_points)
                else:
                    # Circular lane
                    num_points = 50
                    s_vals = np.linspace(0, abs(lane.length), num_points)

                positions = np.array([lane.position(s, 0) for s in s_vals])

                # Plot center line
                ax.plot(positions[:, 0], positions[:, 1], 'k-', linewidth=2, alpha=0.7)

                # Plot lane boundaries
                left_positions = np.array([lane.position(s, lane.width/2) for s in s_vals])
                right_positions = np.array([lane.position(s, -lane.width/2) for s in s_vals])

                ax.plot(left_positions[:, 0], left_positions[:, 1], 'b--', linewidth=1, alpha=0.5)
                ax.plot(right_positions[:, 0], right_positions[:, 1], 'b--', linewidth=1, alpha=0.5)

                # Add segment label
                mid_idx = len(positions) // 2
                mid_pos = positions[mid_idx]
                ax.text(mid_pos[0], mid_pos[1], f"{start_node}→{end_node}",
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Complex Racetrack Layout (loadTrack_03)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('complex_track_layout.png', dpi=150, bbox_inches='tight')
    print("✓ Track layout saved to: complex_track_layout.png")
    plt.show()

    env.close()

def run_test_episode():
    """Run a test episode with visualization."""
    print("\nRunning test episode...")
    env = gym.make("racetrack-complex-v0", render_mode="human")

    obs, info = env.reset()

    print("Controls: Random actions (no trained policy)")
    print("Press Ctrl+C to stop early\n")

    episode_reward = 0
    step_count = 0

    try:
        for _ in range(500):  # Max 500 steps
            # Simple control: try to go forward with slight random steering
            action = np.array([0.5, np.random.uniform(-0.1, 0.1)])  # [acceleration, steering]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Render
            env.render()
            time.sleep(0.05)  # Slow down for visualization

            if terminated or truncated:
                print(f"\nEpisode ended at step {step_count}")
                print(f"  Terminated: {terminated} (crashed or off-road)")
                print(f"  Truncated: {truncated} (max steps)")
                break

        print(f"\nEpisode summary:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Final position: {env.unwrapped.vehicle.position}")
        print(f"  Final speed: {env.unwrapped.vehicle.speed:.2f} m/s")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    env.close()

if __name__ == "__main__":
    # Visualize track layout
    visualize_track_layout()

    # Ask user if they want to run test episode
    response = input("\nDo you want to run a test episode? (y/n): ")
    if response.lower() == 'y':
        run_test_episode()
