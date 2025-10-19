#!/usr/bin/env python3
"""
Test script for DRL-MPCC integration.

This script demonstrates:
1. Pure MPCC mode (baseline)
2. DRL-MPCC integrated mode (with feedforward reference)
3. Comparison of tracking performance and safety
"""

import numpy as np
import gymnasium as gym
import highway_env
import yaml
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from integration.mpcc_drl_wrapper import MPCCDRLWrapper
from integration.drl_mpcc_integration import SafetyAwareDRLMPCC
import tracks.InterpolateTrack as InterpolateTrack


def load_config(config_path="integration/config_integration.yaml"):
    """Load integration configuration."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def extract_track_from_env(env_name="racetrack-single-v0"):
    """Extract track waypoints from highway_env environment."""
    print(f"\n=== Extracting Track from {env_name} ===")

    env = gym.make(env_name)
    obs, info = env.reset()

    # Extract waypoints
    road_network = env.unwrapped.road.network
    lane_sequence = [
        ("a", "b", 0), ("b", "c", 0), ("c", "d", 0), ("d", "e", 0),
        ("e", "f", 0), ("f", "g", 0), ("g", "h", 0), ("h", "i", 0), ("i", "a", 0)
    ]

    waypoints = []
    samples_per_lane = 15

    for lane_id in lane_sequence:
        lane = road_network.get_lane(lane_id)
        for i in range(samples_per_lane):
            longitudinal = i * lane.length / (samples_per_lane - 1) if samples_per_lane > 1 else 0
            position = lane.position(longitudinal, 0)
            waypoints.append([position[0], position[1]])

    waypoints = np.array(waypoints)

    # Remove duplicates
    if len(waypoints) > 1:
        diff = np.diff(waypoints, axis=0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        keep_indices = [0]
        for i in range(1, len(waypoints)):
            if distances[i-1] > 0.8:
                keep_indices.append(i)
        waypoints = waypoints[keep_indices]

    print(f"Extracted {len(waypoints)} waypoints")

    # Save track
    track_name = "rl_single"
    csv_path = f"tracks/{track_name}.csv"
    np.savetxt(csv_path, waypoints, delimiter=", ", fmt="%.15e")

    # Generate lookup table
    r = 2.5
    track_lu_table, smax = InterpolateTrack.generatelookuptable(f"tracks/{track_name}", r)

    env.close()

    return track_name, track_lu_table, smax, r


def run_pure_mpcc(track, vehicleparams, mpccparams, startidx, vx0, Nsim):
    """Run baseline pure MPCC controller."""
    print("\n=== Running Pure MPCC (Baseline) ===")

    # Create pure MPCC controller
    mpcc_params_copy = mpccparams.copy()
    mpcc_params_copy["generate_solver"] = False  # Load existing solver

    mpcc = MPCCDRLWrapper(
        track=track,
        Tsim=mpccparams["Tsim"],
        vehicleparams=vehicleparams,
        mpccparams=mpcc_params_copy,
        control_mode="pure_mpcc"
    )

    # Initial state
    trackvars = ["sval", "tval", "xtrack", "ytrack", "phitrack", "cos(phi)", "sin(phi)", "g_upper", "g_lower"]
    track_lu_table = track["track_lu_table"]

    xt0 = track_lu_table[startidx, trackvars.index("xtrack")]
    yt0 = track_lu_table[startidx, trackvars.index("ytrack")]
    phit0 = track_lu_table[startidx, trackvars.index("phitrack")]
    theta_hat0 = track_lu_table[startidx, trackvars.index("sval")]

    xinit = np.array([xt0, yt0, phit0, vx0, 0.0, 0, 0, 0, theta_hat0])

    # Initialize
    z_current = mpcc.initialize_trajectory(xinit, None, startidx)

    # Simulate
    trajectory = []
    for simidx in range(Nsim):
        z_current = mpcc.update(None)
        ego_pos = z_current[0, 3:5]
        trajectory.append(ego_pos.copy())

        if simidx % 50 == 0:
            speed = z_current[0, mpcc.zvars.index("vx")]
            print(f"  Step {simidx:3d}: speed={speed:.2f} m/s")

    print("Pure MPCC simulation completed!")

    return np.array(trajectory), mpcc


def run_drl_mpcc_integrated(track, vehicleparams, mpccparams, config, startidx, vx0, Nsim):
    """Run DRL-MPCC integrated controller."""
    print("\n=== Running DRL-MPCC Integrated Mode ===")

    # Initial state for MPCC
    trackvars = ["sval", "tval", "xtrack", "ytrack", "phitrack", "cos(phi)", "sin(phi)", "g_upper", "g_lower"]
    track_lu_table = track["track_lu_table"]

    xt0 = track_lu_table[startidx, trackvars.index("xtrack")]
    yt0 = track_lu_table[startidx, trackvars.index("ytrack")]
    phit0 = track_lu_table[startidx, trackvars.index("phitrack")]
    theta_hat0 = track_lu_table[startidx, trackvars.index("sval")]

    xinit = np.array([xt0, yt0, phit0, vx0, 0.0, 0, 0, 0, theta_hat0])

    print(f"Initial position: ({xt0:.2f}, {yt0:.2f}), heading: {phit0:.4f} rad, speed: {vx0:.2f} m/s")

    # Create highway_env environment and set initial state to match MPCC
    env = gym.make("racetrack-single-v0")
    obs, info = env.reset()

    # Manually set highway_env vehicle to match MPCC initial state
    env.unwrapped.vehicle.position = np.array([xt0, yt0])
    env.unwrapped.vehicle.heading = phit0
    env.unwrapped.vehicle.speed = vx0

    print(f"Highway-env observation shape: {obs.shape}")
    print(f"Highway-env vehicle initial pos: ({env.unwrapped.vehicle.position[0]:.2f}, {env.unwrapped.vehicle.position[1]:.2f})")

    # Create DRL-enabled MPCC controller
    mpcc_params_copy = mpccparams.copy()
    mpcc_params_copy["generate_solver"] = config['solver']['generate_solver']

    mpcc = MPCCDRLWrapper(
        track=track,
        Tsim=mpccparams["Tsim"],
        vehicleparams=vehicleparams,
        mpccparams=mpcc_params_copy,
        control_mode="drl_mpcc"
    )

    # Load trained DRL policy
    drl_model_path = config['drl_policy']['model_path']
    try:
        from stable_baselines3 import PPO
        print(f"Loading DRL policy from: {drl_model_path}")
        drl_policy = PPO.load(drl_model_path)
        print("✓ DRL policy loaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to load DRL policy: {e}")
        print("⚠️  Running without DRL (u_ref will be zero)")
        drl_policy = None

    # Create integrated controller
    integration_config = config['integration']
    integrated_ctrl = SafetyAwareDRLMPCC(
        mpcc_controller=mpcc,
        drl_policy=drl_policy,  # Use loaded DRL policy
        alpha=integration_config['alpha'],
        beta=integration_config['beta'],
        K_a=integration_config['K_a'],
        K_delta=integration_config['K_delta']
    )

    # Initialize MPCC
    z_current = mpcc.initialize_trajectory(xinit, None, startidx)

    # Simulate with DRL reference
    trajectory = []
    use_drl_policy = drl_policy is not None

    for simidx in range(Nsim):
        # Get DRL action for current observation
        if use_drl_policy:
            action, _ = drl_policy.predict(obs, deterministic=True)
            # Map to physical units for MPCC reference
            a_ref = 5.0 * float(action[0])
            delta_ref = (np.pi / 4) * float(action[1])
            u_ref = (a_ref, delta_ref)
        else:
            action = np.array([0.0, 0.0])
            u_ref = (0.0, 0.0)

        # Let DRL control the highway_env vehicle
        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            obs, info = env.reset()

        # Update MPCC with the SAME reference that DRL used
        # Don't call integrated_ctrl.step() which would query DRL again
        omega_weights = integrated_ctrl.compute_reference_weights()
        integrated_ctrl.mpcc.u_ref = u_ref
        integrated_ctrl.mpcc.omega_weights = omega_weights
        integrated_ctrl.mpcc.K_ref = np.array([integrated_ctrl.K_a, integrated_ctrl.K_delta])
        integrated_ctrl.mpcc.beta = integrated_ctrl.beta

        # Store for tracking error
        integrated_ctrl.u_ref_history.append([u_ref[0], u_ref[1]])

        # Update MPCC
        z_current = integrated_ctrl.mpcc.update(None)

        # Log actual control
        a_actual = z_current[0, integrated_ctrl.mpcc.zvars.index("a")]
        delta_actual = z_current[0, integrated_ctrl.mpcc.zvars.index("delta")]
        integrated_ctrl.u_actual_history.append([a_actual, delta_actual])

        result = {
            'z_current': z_current,
            'u_ref': u_ref,
            'u_actual': (a_actual, delta_actual)
        }

        z_current = result['z_current']
        ego_pos = z_current[0, 3:5]
        trajectory.append(ego_pos.copy())

        if simidx % 50 == 0:
            speed = z_current[0, mpcc.zvars.index("vx")]
            mpcc_x = z_current[0, mpcc.zvars.index("posx")]
            mpcc_y = z_current[0, mpcc.zvars.index("posy")]
            env_x = env.unwrapped.vehicle.position[0]
            env_y = env.unwrapped.vehicle.position[1]
            env_speed = env.unwrapped.vehicle.speed
            a_ref, delta_ref = result['u_ref']
            print(f"  Step {simidx:3d}: MPCC speed={speed:.2f} m/s, pos=({mpcc_x:.1f},{mpcc_y:.1f}) | Env speed={env_speed:.2f} m/s, pos=({env_x:.1f},{env_y:.1f}) | u_ref=({a_ref:.2f}, {delta_ref:.4f})")

    print("DRL-MPCC integrated simulation completed!")

    # Print tracking statistics
    tracking_stats = integrated_ctrl.get_tracking_error()
    print(f"\nTracking Error Statistics:")
    print(f"  RMSE acceleration: {tracking_stats['rmse_a']:.4f} m/s²")
    print(f"  RMSE steering: {tracking_stats['rmse_delta']:.4f} rad")
    print(f"  Max error accel: {tracking_stats['max_error_a']:.4f} m/s²")
    print(f"  Max error steering: {tracking_stats['max_error_delta']:.4f} rad")

    env.close()
    return np.array(trajectory), mpcc, integrated_ctrl


def plot_comparison(track, traj_pure, traj_drl):
    """Plot trajectory comparison."""
    print("\n=== Generating Comparison Plot ===")

    trackvars = ["sval", "tval", "xtrack", "ytrack", "phitrack", "cos(phi)", "sin(phi)", "g_upper", "g_lower"]
    track_lu_table = track["track_lu_table"]

    plt.figure(figsize=(14, 10))

    # Plot track
    plt.plot(track_lu_table[:, trackvars.index("xtrack")],
             track_lu_table[:, trackvars.index("ytrack")],
             'k--', linewidth=2, alpha=0.5, label='Reference Track')

    # Plot pure MPCC
    plt.plot(traj_pure[:, 0], traj_pure[:, 1],
             'b-', linewidth=2, label='Pure MPCC', alpha=0.7)
    plt.plot(traj_pure[0, 0], traj_pure[0, 1],
             'go', markersize=10, label='Start')

    # Plot DRL-MPCC
    plt.plot(traj_drl[:, 0], traj_drl[:, 1],
             'r-', linewidth=2, label='DRL-MPCC Integrated', alpha=0.7)

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Y [m]', fontsize=12)
    plt.title('DRL-MPCC Integration: Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)

    # Save plot
    os.makedirs('integration/results', exist_ok=True)
    plt.savefig('integration/results/trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: integration/results/trajectory_comparison.png")
    plt.close()


def main():
    """Main test function."""
    print("=" * 70)
    print("DRL-MPCC INTEGRATION TEST")
    print("=" * 70)

    # Load configuration
    config = load_config()
    print(f"\nControl Mode: {config['control_mode']}")

    # Load parameters
    with open("config/vehicleparams.yaml") as f:
        vehicleparams = yaml.load(f, Loader=yaml.FullLoader)

    with open("config/mpccparams.yaml") as f:
        mpccparams = yaml.load(f, Loader=yaml.FullLoader)

    # Extract track
    track_name, track_lu_table, smax, r = extract_track_from_env()

    # Load track params
    with open(f"tracks/{track_name}_params.yaml") as f:
        track_params = yaml.load(f, Loader=yaml.FullLoader)
    ppm = track_params["ppm"]

    track = {
        "track_lu_table": track_lu_table,
        "smax": smax,
        "r": r,
        "ppm": ppm
    }

    # Simulation parameters
    startidx = 10
    vx0 = 5.0
    Nsim = 800  # Increased for longer simulation (covers more of the track)

    # Run both modes
    traj_pure, mpcc_pure = run_pure_mpcc(track, vehicleparams, mpccparams, startidx, vx0, Nsim)

    traj_drl, mpcc_drl, integrated_ctrl = run_drl_mpcc_integrated(
        track, vehicleparams, mpccparams, config, startidx, vx0, Nsim
    )

    # Compare results
    plot_comparison(track, traj_pure, traj_drl)

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
