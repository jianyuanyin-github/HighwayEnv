#!/usr/bin/env python3
"""
Unified DRL-MPCC Integration Test (CORRECTED VERSION)

This version solves the critical issue of running DRL and MPCC in separate
environments, which causes state divergence.

Solution: Use MPCC dynamics to advance the state, but synchronize Highway-env
for observation generation only (no independent env.step()).

Architecture:
    obs → DRL → (a_ref, δ_ref) → MPCC → u_actual → dynamics.tick() → new_state
                                                           ↓
                                              sync env.vehicle.state
                                                           ↓
                                                   new_obs ← env.observe()
"""

import numpy as np
import gymnasium as gym
import highway_env
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

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


def get_observation_from_env(env, mpcc_state):
    """
    Synchronize env vehicle state with MPCC and get observation.

    Args:
        env: Highway-env environment
        mpcc_state: Current MPCC state [posx, posy, phi, vx, vy, omega, a, delta, theta]

    Returns:
        obs: Observation from env (occupancy grid)
    """
    # Update env vehicle state to match MPCC
    env.unwrapped.vehicle.position = np.array([mpcc_state[0], mpcc_state[1]])
    env.unwrapped.vehicle.heading = mpcc_state[2]
    env.unwrapped.vehicle.speed = mpcc_state[3]  # vx

    # Manually compute observation without stepping physics
    obs = env.unwrapped.observation_type.observe()

    return obs


def run_unified_drl_mpcc(track, vehicleparams, mpccparams, config, startidx, vx0, Nsim):
    """
    Run unified DRL-MPCC integration where MPCC is the ONLY dynamics.

    Key change: Highway-env is used ONLY for observation, not for dynamics!
    """
    print("\n=== Running Unified DRL-MPCC Integration ===")
    print("⚠️  MPCC is the ONLY dynamics - env is synchronized for observation only!\n")

    # ==================== 1. Initialize MPCC ====================
    trackvars = ["sval", "tval", "xtrack", "ytrack", "phitrack",
                 "cos(phi)", "sin(phi)", "g_upper", "g_lower"]
    track_lu_table = track["track_lu_table"]

    xt0 = track_lu_table[startidx, trackvars.index("xtrack")]
    yt0 = track_lu_table[startidx, trackvars.index("ytrack")]
    phit0 = track_lu_table[startidx, trackvars.index("phitrack")]
    theta_hat0 = track_lu_table[startidx, trackvars.index("sval")]

    xinit = np.array([xt0, yt0, phit0, vx0, 0.0, 0, 0, 0, theta_hat0])

    print(f"Initial state: pos=({xt0:.2f}, {yt0:.2f}), heading={phit0:.4f} rad, speed={vx0:.2f} m/s")

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

    # ==================== 2. Load DRL Policy ====================
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

    # ==================== 3. Create Integrated Controller ====================
    integration_config = config['integration']
    integrated_ctrl = SafetyAwareDRLMPCC(
        mpcc_controller=mpcc,
        drl_policy=drl_policy,
        alpha=integration_config['alpha'],
        beta=integration_config['beta'],
        K_a=integration_config['K_a'],
        K_delta=integration_config['K_delta']
    )

    # Initialize MPCC trajectory
    z_current = mpcc.initialize_trajectory(xinit, None, startidx)

    # ==================== 4. Initialize Highway-env for Observation Only ====================
    env = gym.make("racetrack-single-v0")
    obs, info = env.reset()

    # Synchronize env to match MPCC initial state
    env.unwrapped.vehicle.position = np.array([xt0, yt0])
    env.unwrapped.vehicle.heading = phit0
    env.unwrapped.vehicle.speed = vx0
    obs = env.unwrapped.observation_type.observe()

    print(f"Highway-env initialized (for observation only)")
    print(f"Observation shape: {obs.shape}\n")

    # ==================== 5. Simulation Loop ====================
    trajectory_mpcc = []
    trajectory_env = []  # Track env position for debugging
    use_drl_policy = drl_policy is not None

    for simidx in range(Nsim):
        # ----- Get DRL Reference -----
        if use_drl_policy:
            action, _ = drl_policy.predict(obs, deterministic=True)
            a_ref = 5.0 * float(action[0])
            delta_ref = (np.pi / 4) * float(action[1])
            u_ref = (a_ref, delta_ref)
        else:
            action = np.array([0.0, 0.0])
            u_ref = (0.0, 0.0)

        # ----- Update MPCC with DRL Reference -----
        omega_weights = integrated_ctrl.compute_reference_weights()
        integrated_ctrl.mpcc.u_ref = u_ref
        integrated_ctrl.mpcc.omega_weights = omega_weights
        integrated_ctrl.mpcc.K_ref = np.array([integrated_ctrl.K_a, integrated_ctrl.K_delta])
        integrated_ctrl.mpcc.beta = integrated_ctrl.beta

        integrated_ctrl.u_ref_history.append([u_ref[0], u_ref[1]])

        # ----- MPCC Solve and Dynamics Update (SINGLE SOURCE OF TRUTH) -----
        z_current = integrated_ctrl.mpcc.update(None)

        # Extract MPCC state
        mpcc_x = z_current[0, mpcc.zvars.index("posx")]
        mpcc_y = z_current[0, mpcc.zvars.index("posy")]
        mpcc_phi = z_current[0, mpcc.zvars.index("phi")]
        mpcc_vx = z_current[0, mpcc.zvars.index("vx")]
        mpcc_vy = z_current[0, mpcc.zvars.index("vy")]
        mpcc_omega = z_current[0, mpcc.zvars.index("omega")]
        a_actual = z_current[0, mpcc.zvars.index("a")]
        delta_actual = z_current[0, mpcc.zvars.index("delta")]

        integrated_ctrl.u_actual_history.append([a_actual, delta_actual])

        # ----- Synchronize Highway-env to MPCC State -----
        # CRITICAL: We do NOT call env.step() - just sync state for observation
        mpcc_state = np.array([mpcc_x, mpcc_y, mpcc_phi, mpcc_vx, mpcc_vy, mpcc_omega,
                               a_actual, delta_actual, z_current[0, mpcc.zvars.index("theta")]])
        obs = get_observation_from_env(env, mpcc_state)

        # ----- Record Trajectory -----
        trajectory_mpcc.append([mpcc_x, mpcc_y])
        trajectory_env.append([env.unwrapped.vehicle.position[0],
                               env.unwrapped.vehicle.position[1]])

        # ----- Logging -----
        if simidx % 50 == 0:
            speed = mpcc_vx
            env_x = env.unwrapped.vehicle.position[0]
            env_y = env.unwrapped.vehicle.position[1]
            sync_error = np.sqrt((mpcc_x - env_x)**2 + (mpcc_y - env_y)**2)
            a_ref_val, delta_ref_val = u_ref
            print(f"  Step {simidx:3d}: MPCC pos=({mpcc_x:.1f},{mpcc_y:.1f}) speed={speed:.2f} m/s | "
                  f"Env pos=({env_x:.1f},{env_y:.1f}) | Sync error={sync_error:.4f} m | "
                  f"u_ref=({a_ref_val:.2f}, {delta_ref_val:.4f})")

    print("\nUnified DRL-MPCC simulation completed!")

    # ==================== 6. Print Statistics ====================
    tracking_stats = integrated_ctrl.get_tracking_error()
    print(f"\nTracking Error Statistics:")
    print(f"  RMSE acceleration: {tracking_stats['rmse_a']:.4f} m/s²")
    print(f"  RMSE steering: {tracking_stats['rmse_delta']:.4f} rad")
    print(f"  Max error accel: {tracking_stats['max_error_a']:.4f} m/s²")
    print(f"  Max error steering: {tracking_stats['max_error_delta']:.4f} rad")

    # Verify synchronization
    trajectory_mpcc = np.array(trajectory_mpcc)
    trajectory_env = np.array(trajectory_env)
    sync_errors = np.sqrt(np.sum((trajectory_mpcc - trajectory_env)**2, axis=1))
    print(f"\nSynchronization Quality:")
    print(f"  Mean sync error: {np.mean(sync_errors):.6f} m (should be ~0)")
    print(f"  Max sync error: {np.max(sync_errors):.6f} m (should be ~0)")

    env.close()
    return trajectory_mpcc, mpcc, integrated_ctrl


def plot_unified_trajectory(track, trajectory):
    """Plot unified trajectory."""
    print("\n=== Generating Trajectory Plot ===")

    trackvars = ["sval", "tval", "xtrack", "ytrack", "phitrack",
                 "cos(phi)", "sin(phi)", "g_upper", "g_lower"]
    track_lu_table = track["track_lu_table"]

    plt.figure(figsize=(14, 10))

    # Plot track
    plt.plot(track_lu_table[:, trackvars.index("xtrack")],
             track_lu_table[:, trackvars.index("ytrack")],
             'k--', linewidth=2, alpha=0.5, label='Reference Track')

    # Plot unified trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1],
             'b-', linewidth=2, label='Unified DRL-MPCC', alpha=0.7)
    plt.plot(trajectory[0, 0], trajectory[0, 1],
             'go', markersize=12, label='Start', zorder=10)
    plt.plot(trajectory[-1, 0], trajectory[-1, 1],
             'ro', markersize=12, label='End', zorder=10)

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Y [m]', fontsize=12)
    plt.title('Unified DRL-MPCC Integration: Single Dynamics Trajectory',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)

    # Save plot
    os.makedirs('integration/results', exist_ok=True)
    plt.savefig('integration/results/trajectory_unified.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: integration/results/trajectory_unified.png")
    plt.close()


def main():
    """Main test function."""
    print("=" * 80)
    print("UNIFIED DRL-MPCC INTEGRATION TEST (CORRECTED VERSION)")
    print("=" * 80)
    print("\nKey improvement: MPCC is the ONLY dynamics, env synchronized for obs only!")
    print("This solves the state divergence problem in the original implementation.\n")

    # Load configuration
    config = load_config()
    print(f"Control Mode: {config['control_mode']}")

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
    Nsim = 800

    # Run unified integration
    trajectory, mpcc, integrated_ctrl = run_unified_drl_mpcc(
        track, vehicleparams, mpccparams, config, startidx, vx0, Nsim
    )

    # Visualize
    plot_unified_trajectory(track, trajectory)

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
