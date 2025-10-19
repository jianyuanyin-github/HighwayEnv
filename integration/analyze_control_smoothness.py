#!/usr/bin/env python3
"""
Analyze control smoothness to check if DRL oscillation causes performance degradation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import yaml
import gymnasium as gym
import highway_env
import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from integration.mpcc_drl_wrapper import MPCCDRLWrapper
from integration.drl_mpcc_integration import SafetyAwareDRLMPCC
import tracks.InterpolateTrack as InterpolateTrack


def extract_track():
    """Extract track from environment."""
    env = gym.make("racetrack-single-v0")
    obs, info = env.reset()

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

    if len(waypoints) > 1:
        diff = np.diff(waypoints, axis=0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        keep_indices = [0]
        for i in range(1, len(waypoints)):
            if distances[i-1] > 0.8:
                keep_indices.append(i)
        waypoints = waypoints[keep_indices]

    track_name = "rl_single"
    csv_path = f"tracks/{track_name}.csv"
    np.savetxt(csv_path, waypoints, delimiter=", ", fmt="%.15e")

    r = 2.5
    track_lu_table, smax = InterpolateTrack.generatelookuptable(f"tracks/{track_name}", r)

    with open(f"tracks/{track_name}_params.yaml") as f:
        track_params = yaml.load(f, Loader=yaml.FullLoader)
    ppm = track_params["ppm"]

    env.close()

    return {
        "track_lu_table": track_lu_table,
        "smax": smax,
        "r": r,
        "ppm": ppm
    }


def run_pure_drl(track, Nsim=300):
    """Run pure DRL to extract control sequence."""
    print("\n=== Running Pure DRL (for comparison) ===")

    env = gym.make("racetrack-single-v0")
    obs, info = env.reset()

    # Load DRL policy
    from stable_baselines3 import PPO
    drl_policy = PPO.load("racetrack_single_ppo/model_1760367700.zip")

    drl_actions = []

    for step in range(Nsim):
        action, _ = drl_policy.predict(obs, deterministic=True)

        # Convert to physical units
        a_ref = 5.0 * float(action[0])
        delta_ref = (np.pi / 4) * float(action[1])

        drl_actions.append([a_ref, delta_ref])

        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            obs, info = env.reset()

    env.close()
    return np.array(drl_actions)


def run_drl_mpcc_integrated(track, vehicleparams, mpccparams, config, Nsim=300):
    """Run DRL-MPCC integration and record control."""
    print("\n=== Running DRL-MPCC Integration ===")

    trackvars = ["sval", "tval", "xtrack", "ytrack", "phitrack",
                 "cos(phi)", "sin(phi)", "g_upper", "g_lower"]
    track_lu_table = track["track_lu_table"]

    startidx = 10
    vx0 = 5.0

    xt0 = track_lu_table[startidx, trackvars.index("xtrack")]
    yt0 = track_lu_table[startidx, trackvars.index("ytrack")]
    phit0 = track_lu_table[startidx, trackvars.index("phitrack")]
    theta_hat0 = track_lu_table[startidx, trackvars.index("sval")]

    xinit = np.array([xt0, yt0, phit0, vx0, 0.0, 0, 0, 0, theta_hat0])

    # Create MPCC controller
    mpcc_params_copy = mpccparams.copy()
    mpcc_params_copy["generate_solver"] = False

    mpcc = MPCCDRLWrapper(
        track=track,
        Tsim=mpccparams["Tsim"],
        vehicleparams=vehicleparams,
        mpccparams=mpcc_params_copy,
        control_mode="drl_mpcc"
    )

    # Load DRL policy
    from stable_baselines3 import PPO
    drl_policy = PPO.load(config['drl_policy']['model_path'])

    # Create integrated controller
    integration_config = config['integration']
    integrated_ctrl = SafetyAwareDRLMPCC(
        mpcc_controller=mpcc,
        drl_policy=drl_policy,
        alpha=integration_config['alpha'],
        beta=integration_config['beta'],
        K_a=integration_config['K_a'],
        K_delta=integration_config['K_delta']
    )

    # Initialize env
    env = gym.make("racetrack-single-v0")
    obs, info = env.reset()
    env.unwrapped.vehicle.position = np.array([xt0, yt0])
    env.unwrapped.vehicle.heading = phit0
    env.unwrapped.vehicle.speed = vx0
    obs = env.unwrapped.observation_type.observe()

    # Initialize MPCC
    z_current = mpcc.initialize_trajectory(xinit, None, startidx)

    u_refs = []
    u_actuals = []

    for simidx in range(Nsim):
        # Get DRL reference
        action, _ = drl_policy.predict(obs, deterministic=True)
        a_ref = 5.0 * float(action[0])
        delta_ref = (np.pi / 4) * float(action[1])
        u_ref = (a_ref, delta_ref)

        # Update MPCC
        omega_weights = integrated_ctrl.compute_reference_weights()
        integrated_ctrl.mpcc.u_ref = u_ref
        integrated_ctrl.mpcc.omega_weights = omega_weights
        integrated_ctrl.mpcc.K_ref = np.array([integrated_ctrl.K_a, integrated_ctrl.K_delta])
        integrated_ctrl.mpcc.beta = integrated_ctrl.beta

        z_current = integrated_ctrl.mpcc.update(None)

        # Extract actual control
        a_actual = z_current[0, mpcc.zvars.index("a")]
        delta_actual = z_current[0, mpcc.zvars.index("delta")]

        u_refs.append([a_ref, delta_ref])
        u_actuals.append([a_actual, delta_actual])

        # Sync env for observation
        mpcc_x = z_current[0, mpcc.zvars.index("posx")]
        mpcc_y = z_current[0, mpcc.zvars.index("posy")]
        mpcc_phi = z_current[0, mpcc.zvars.index("phi")]
        mpcc_vx = z_current[0, mpcc.zvars.index("vx")]

        env.unwrapped.vehicle.position = np.array([mpcc_x, mpcc_y])
        env.unwrapped.vehicle.heading = mpcc_phi
        env.unwrapped.vehicle.speed = mpcc_vx
        obs = env.unwrapped.observation_type.observe()

    env.close()
    return np.array(u_refs), np.array(u_actuals)


def analyze_smoothness(u_sequence, label):
    """Analyze control smoothness."""
    a_seq = u_sequence[:, 0]
    delta_seq = u_sequence[:, 1]

    # Compute differences (jerk and steering rate)
    a_diff = np.diff(a_seq)
    delta_diff = np.diff(delta_seq)

    # Statistics
    stats = {
        "label": label,
        "a_mean": np.mean(a_seq),
        "a_std": np.std(a_seq),
        "a_jerk_mean": np.mean(np.abs(a_diff)),
        "a_jerk_std": np.std(a_diff),
        "delta_mean": np.mean(delta_seq),
        "delta_std": np.std(delta_seq),
        "delta_rate_mean": np.mean(np.abs(delta_diff)),
        "delta_rate_std": np.std(delta_diff),
    }

    return stats


def plot_control_comparison(drl_actions, u_refs, u_actuals):
    """Plot control sequences for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    steps = len(u_actuals)
    time = np.arange(steps) * 0.05  # 20 Hz policy frequency

    # Acceleration
    axes[0, 0].plot(time, drl_actions[:steps, 0], 'g-', alpha=0.7, linewidth=2, label='Pure DRL')
    axes[0, 0].plot(time, u_refs[:, 0], 'b--', alpha=0.7, linewidth=1.5, label='DRL Reference')
    axes[0, 0].plot(time, u_actuals[:, 0], 'r-', alpha=0.8, linewidth=2, label='MPCC Actual')
    axes[0, 0].set_xlabel('Time [s]', fontsize=12)
    axes[0, 0].set_ylabel('Acceleration [m/s²]', fontsize=12)
    axes[0, 0].set_title('Acceleration Control', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Steering
    axes[0, 1].plot(time, drl_actions[:steps, 1], 'g-', alpha=0.7, linewidth=2, label='Pure DRL')
    axes[0, 1].plot(time, u_refs[:, 1], 'b--', alpha=0.7, linewidth=1.5, label='DRL Reference')
    axes[0, 1].plot(time, u_actuals[:, 1], 'r-', alpha=0.8, linewidth=2, label='MPCC Actual')
    axes[0, 1].set_xlabel('Time [s]', fontsize=12)
    axes[0, 1].set_ylabel('Steering Angle [rad]', fontsize=12)
    axes[0, 1].set_title('Steering Control', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Acceleration rate (Jerk)
    a_jerk_drl = np.diff(drl_actions[:steps, 0]) / 0.05
    a_jerk_ref = np.diff(u_refs[:, 0]) / 0.05
    a_jerk_actual = np.diff(u_actuals[:, 0]) / 0.05

    axes[1, 0].plot(time[:-1], a_jerk_drl, 'g-', alpha=0.7, linewidth=2, label='Pure DRL')
    axes[1, 0].plot(time[:-1], a_jerk_ref, 'b--', alpha=0.7, linewidth=1.5, label='DRL Reference')
    axes[1, 0].plot(time[:-1], a_jerk_actual, 'r-', alpha=0.8, linewidth=2, label='MPCC Actual')
    axes[1, 0].set_xlabel('Time [s]', fontsize=12)
    axes[1, 0].set_ylabel('Jerk [m/s³]', fontsize=12)
    axes[1, 0].set_title('Acceleration Rate (Jerk)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Steering rate
    delta_rate_drl = np.diff(drl_actions[:steps, 1]) / 0.05
    delta_rate_ref = np.diff(u_refs[:, 1]) / 0.05
    delta_rate_actual = np.diff(u_actuals[:, 1]) / 0.05

    axes[1, 1].plot(time[:-1], delta_rate_drl, 'g-', alpha=0.7, linewidth=2, label='Pure DRL')
    axes[1, 1].plot(time[:-1], delta_rate_ref, 'b--', alpha=0.7, linewidth=1.5, label='DRL Reference')
    axes[1, 1].plot(time[:-1], delta_rate_actual, 'r-', alpha=0.8, linewidth=2, label='MPCC Actual')
    axes[1, 1].set_xlabel('Time [s]', fontsize=12)
    axes[1, 1].set_ylabel('Steering Rate [rad/s]', fontsize=12)
    axes[1, 1].set_title('Steering Rate', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('integration/results', exist_ok=True)
    plt.savefig('integration/results/control_smoothness_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to: integration/results/control_smoothness_analysis.png")
    plt.close()


def main():
    print("=" * 80)
    print("CONTROL SMOOTHNESS ANALYSIS")
    print("=" * 80)

    # Load configs
    with open("integration/config_integration.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open("config/vehicleparams.yaml") as f:
        vehicleparams = yaml.load(f, Loader=yaml.FullLoader)

    with open("config/mpccparams.yaml") as f:
        mpccparams = yaml.load(f, Loader=yaml.FullLoader)

    # Extract track
    print("\n=== Extracting Track ===")
    track = extract_track()

    # Run tests
    Nsim = 300

    drl_actions = run_pure_drl(track, Nsim)
    u_refs, u_actuals = run_drl_mpcc_integrated(track, vehicleparams, mpccparams, config, Nsim)

    # Analyze smoothness
    print("\n" + "=" * 80)
    print("SMOOTHNESS STATISTICS")
    print("=" * 80)

    stats_drl = analyze_smoothness(drl_actions, "Pure DRL")
    stats_ref = analyze_smoothness(u_refs, "DRL Reference")
    stats_actual = analyze_smoothness(u_actuals, "MPCC Actual")

    print(f"\n{'Metric':<30} {'Pure DRL':>15} {'DRL Ref':>15} {'MPCC Actual':>15}")
    print("-" * 80)
    print(f"{'Accel Mean [m/s²]':<30} {stats_drl['a_mean']:>15.3f} {stats_ref['a_mean']:>15.3f} {stats_actual['a_mean']:>15.3f}")
    print(f"{'Accel Std [m/s²]':<30} {stats_drl['a_std']:>15.3f} {stats_ref['a_std']:>15.3f} {stats_actual['a_std']:>15.3f}")
    print(f"{'Jerk Mean [m/s³]':<30} {stats_drl['a_jerk_mean']:>15.3f} {stats_ref['a_jerk_mean']:>15.3f} {stats_actual['a_jerk_mean']:>15.3f}")
    print(f"{'Jerk Std [m/s³]':<30} {stats_drl['a_jerk_std']:>15.3f} {stats_ref['a_jerk_std']:>15.3f} {stats_actual['a_jerk_std']:>15.3f}")
    print()
    print(f"{'Steering Mean [rad]':<30} {stats_drl['delta_mean']:>15.3f} {stats_ref['delta_mean']:>15.3f} {stats_actual['delta_mean']:>15.3f}")
    print(f"{'Steering Std [rad]':<30} {stats_drl['delta_std']:>15.3f} {stats_ref['delta_std']:>15.3f} {stats_actual['delta_std']:>15.3f}")
    print(f"{'Steering Rate Mean [rad/s]':<30} {stats_drl['delta_rate_mean']:>15.3f} {stats_ref['delta_rate_mean']:>15.3f} {stats_actual['delta_rate_mean']:>15.3f}")
    print(f"{'Steering Rate Std [rad/s]':<30} {stats_drl['delta_rate_std']:>15.3f} {stats_ref['delta_rate_std']:>15.3f} {stats_actual['delta_rate_std']:>15.3f}")

    # Plot
    plot_control_comparison(drl_actions, u_refs, u_actuals)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
