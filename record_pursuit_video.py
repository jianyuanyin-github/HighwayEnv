#!/usr/bin/env python3
"""
Record Pure Pursuit racetrack environment as video.
"""

import gymnasium as gym
import highway_env
import numpy as np

def main():
    # Create Pure Pursuit environment with video recording
    env = gym.make(
        'racetrack-single-pursuit-v0',
        render_mode='rgb_array'
    )

    # Wrap with RecordVideo
    from gymnasium.wrappers import RecordVideo
    env = RecordVideo(
        env,
        video_folder='racetrack_single_ppo/videos_pursuit',
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix='pure_pursuit'
    )

    print('='*60)
    print('Recording Pure Pursuit Video')
    print('='*60)
    print()

    obs, info = env.reset()

    print(f'Vehicle type: {type(env.unwrapped.vehicle).__name__}')
    print(f'Recording to: racetrack_single_ppo/videos_pursuit/')
    print()

    total_reward = 0
    step = 0

    # Simple strategy: adjust speed based on lane type
    def get_action(vehicle):
        """Adjust speed based on lane curvature."""
        lane = vehicle.lane
        lane_type = type(lane).__name__

        if 'Straight' in lane_type:
            return 2  # FASTER
        else:  # CircularLane
            if hasattr(lane, 'radius'):
                if lane.radius < 20:
                    return 0  # SLOWER (tight curve)
                else:
                    return 1  # IDLE (medium curve)
            else:
                return 1

    # Run for one full episode
    while True:
        action = get_action(env.unwrapped.vehicle)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 100 == 0:
            vehicle = env.unwrapped.vehicle
            lane_idx = vehicle.lane_index
            print(f'Step {step:4d}: {lane_idx[0]}→{lane_idx[1]}, '
                  f'speed={vehicle.speed:5.2f}, reward={total_reward:6.1f}')

        if done or truncated:
            print()
            print(f'Episode finished at step {step}')
            print(f'Total reward: {total_reward:.2f}')
            break

    env.close()
    print()
    print('Video saved to: racetrack_single_ppo/videos_pursuit/')
    print('Done!')


if __name__ == '__main__':
    main()
