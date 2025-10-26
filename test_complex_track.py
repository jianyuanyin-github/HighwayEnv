#!/usr/bin/env python3
"""
Test script for the new complex track configuration.
"""
import gymnasium as gym
import numpy as np
import highway_env

def test_track():
    """Test the complex track environment."""
    print("Creating racetrack-complex-v0 environment...")

    try:
        env = gym.make("racetrack-complex-v0", render_mode="rgb_array")
        print("✓ Environment created successfully")

        print("\nResetting environment...")
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        print(f"  Vehicle position: {env.unwrapped.vehicle.position}")
        print(f"  Vehicle heading: {np.rad2deg(env.unwrapped.vehicle.heading):.1f}°")

        # Check road network
        road_network = env.unwrapped.road.network
        print(f"\n✓ Road network created")
        print(f"  Total segments: {len(road_network.graph)}")

        # List all segments
        print("\nTrack segments:")
        for start_node in sorted(road_network.graph.keys()):
            for end_node in sorted(road_network.graph[start_node].keys()):
                lanes = road_network.graph[start_node][end_node]
                for lane_idx, lane in enumerate(lanes):
                    lane_type = type(lane).__name__
                    speed_limit = lane.speed_limit
                    width = lane.width
                    if hasattr(lane, 'length'):
                        length = lane.length
                        print(f"  {start_node} → {end_node}: {lane_type}, "
                              f"length={length:.1f}m, width={width:.1f}m, "
                              f"speed={speed_limit:.1f}m/s")
                    else:
                        print(f"  {start_node} → {end_node}: {lane_type}, "
                              f"width={width:.1f}m, speed={speed_limit:.1f}m/s")

        # Test a few steps
        print("\n✓ Testing simulation steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"  Episode ended at step {i}")
                print(f"    Terminated: {terminated}, Truncated: {truncated}")
                break
        else:
            print(f"  Successfully ran 10 steps")
            print(f"  Vehicle position: {env.unwrapped.vehicle.position}")
            print(f"  Vehicle speed: {env.unwrapped.vehicle.speed:.2f} m/s")

        env.close()
        print("\n✓ All tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_track()
    exit(0 if success else 1)
