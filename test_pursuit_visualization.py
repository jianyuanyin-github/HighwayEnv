#!/usr/bin/env python3
"""
Visualize Pure Pursuit racetrack environment.
"""

import gymnasium as gym
import highway_env
import numpy as np

def main():
    # Create Pure Pursuit environment
    env = gym.make('racetrack-single-pursuit-v0', render_mode='human')

    print('='*60)
    print('Pure Pursuit Racetrack Visualization')
    print('='*60)
    print()
    print('Controls:')
    print('  Agent uses DiscreteMetaAction:')
    print('    Action 0: SLOWER')
    print('    Action 1: IDLE')
    print('    Action 2: FASTER')
    print()
    print('  Steering is AUTOMATIC via Pure Pursuit!')
    print()
    print('Strategy: Speed up on straights, slow down on curves')
    print('='*60)
    print()

    obs, info = env.reset()

    print(f'Vehicle type: {type(env.unwrapped.vehicle).__name__}')
    print(f'Initial position: {env.unwrapped.vehicle.position}')
    print(f'Route: {env.unwrapped.vehicle.route}')
    print()
    print('Starting...')
    print()

    total_reward = 0
    step = 0

    # Simple strategy: adjust speed based on lane type
    def get_action(vehicle):
        """
        Simple strategy:
        - Fast on straight lanes (StraightLane)
        - Slow on circular lanes (CircularLane)

        Actions:
          0: SLOWER
          1: IDLE
          2: FASTER
        """
        lane = vehicle.lane
        lane_type = type(lane).__name__

        if 'Straight' in lane_type:
            return 2  # FASTER
        else:  # CircularLane
            # Slower on tighter curves
            if hasattr(lane, 'radius'):
                if lane.radius < 20:
                    return 0  # SLOWER
                else:
                    return 1  # IDLE
            else:
                return 1  # IDLE

    while True:
        # Get action based on current lane
        action = get_action(env.unwrapped.vehicle)

        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Render
        env.render()

        # Print info every 50 steps
        if step % 50 == 0:
            vehicle = env.unwrapped.vehicle
            lane_idx = vehicle.lane_index
            pos = vehicle.position
            speed = vehicle.speed
            target_speed = vehicle.target_speed
            lane_type = type(vehicle.lane).__name__

            print(f'Step {step:4d}: {lane_idx[0]}→{lane_idx[1]} ({lane_type:12s}), '
                  f'pos=({pos[0]:6.1f},{pos[1]:6.1f}), '
                  f'speed={speed:5.2f}, target={target_speed:4.1f}, '
                  f'reward={total_reward:6.1f}')

        # Check termination
        if done or truncated:
            print()
            print(f'Episode finished at step {step}!')
            print(f'Total reward: {total_reward:.2f}')
            print(f'Reason: done={done}, truncated={truncated}')
            print()
            print('Restarting...')
            print()
            obs, info = env.reset()
            total_reward = 0
            step = 0

    env.close()
    print('\nDone!')


if __name__ == '__main__':
    main()
