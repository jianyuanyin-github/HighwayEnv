#!/usr/bin/env python3
"""
Run MPCC controller directly on RacetrackEnvSingle track
"""

import numpy as np
import gymnasium as gym
import highway_env
import tracks.InterpolateTrack as InterpolateTrack
from MPCC.MPCC_class import MPCC
import yaml
import matplotlib.pyplot as plt


def extract_and_run_mpcc():
    """Extract RL track and run MPCC on it"""
    
    print("=== Step 1: Extract RL Track ===")
    # Create environment to extract track
    env = gym.make("racetrack-single-v0")
    obs, info = env.reset()
    
    # Extract waypoints from road network
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
    
    # Save track
    track_name = "rl_single"
    csv_path = f"tracks/{track_name}.csv"
    np.savetxt(csv_path, waypoints, delimiter=", ", fmt="%.15e")
    print(f"Saved {len(waypoints)} waypoints to {csv_path}")
    
    # Generate lookup table
    r = 2.5
    track_lu_table, smax = InterpolateTrack.generatelookuptable(f"tracks/{track_name}", r)
    print(f"Generated lookup table: smax={smax:.2f}m")
    
    env.close()
    
    print("\n=== Step 2: Run MPCC ===")
    # Load MPCC parameters
    with open("config/vehicleparams.yaml") as file:
        vehicleparams = yaml.load(file, Loader=yaml.FullLoader)
    
    with open("config/mpccparams.yaml") as file:
        mpccparams = yaml.load(file, Loader=yaml.FullLoader)
    
    with open("config/simulationparams.yaml") as file:
        simulationparams = yaml.load(file, Loader=yaml.FullLoader)
    
    # Update simulation params to use our track
    simulationparams["trackname"] = track_name
    
    Tsim = mpccparams["Tsim"]
    Tf = mpccparams["Tf"] 
    N = mpccparams["N"]
    Nsim = int(np.floor(N * Tsim / Tf))
    
    # Create track data structure
    with open(f"tracks/{track_name}_params.yaml") as file:
        track_params = yaml.load(file, Loader=yaml.FullLoader)
    ppm = track_params["ppm"]
    
    track = {
        "track_lu_table": track_lu_table,
        "smax": smax,
        "r": r,
        "ppm": ppm
    }
    
    # Initialize MPCC
    MPCC_Controller = MPCC(track, Tsim, vehicleparams, mpccparams)
    
    # Initial state - start at beginning of track
    trackvars = ["sval", "tval", "xtrack", "ytrack", "phitrack", "cos(phi)", "sin(phi)", "g_upper", "g_lower"]
    startidx = 10  # Start a bit into the track
    
    xt0 = track_lu_table[startidx, trackvars.index("xtrack")]
    yt0 = track_lu_table[startidx, trackvars.index("ytrack")] 
    phit0 = track_lu_table[startidx, trackvars.index("phitrack")]
    theta_hat0 = track_lu_table[startidx, trackvars.index("sval")]
    
    vx0 = 5.0  # Start with 5 m/s
    xinit = np.array([xt0, yt0, phit0, vx0, 0.0, 0, 0, 0, theta_hat0])
    
    print(f"Starting at: x={xt0:.2f}, y={yt0:.2f}, theta={theta_hat0:.2f}")
    
    # Initialize trajectory
    z_current = MPCC_Controller.initialize_trajectory(xinit, None, startidx)
    
    print(f"\n=== Step 3: MPCC Simulation ({Nsim} steps) ===")
    
    # Variables for plotting
    zvars = ["jerk", "deltadot", "thetadot", "posx", "posy", "phi", "vx", "vy", "omega", "a", "delta", "theta"]
    
    # Storage for trajectory
    ego_trajectory = []
    
    # Run simulation
    for simidx in range(Nsim):
        z_current = MPCC_Controller.update(None)
        
        # Store ego vehicle position
        ego_pos = z_current[0, 3:5]  # [x, y]
        ego_trajectory.append(ego_pos.copy())
        
        if simidx % 10 == 0:
            current_speed = z_current[0, zvars.index("vx")]
            current_theta = z_current[0, zvars.index("theta")]
            print(f"Step {simidx:3d}: speed={current_speed:.2f} m/s, theta={current_theta:.2f}")
    
    print("MPCC simulation completed!")
    
    # Plot results
    ego_trajectory = np.array(ego_trajectory)
    
    plt.figure(figsize=(12, 8))
    
    # Plot track
    plt.plot(track_lu_table[:, trackvars.index("xtrack")], 
             track_lu_table[:, trackvars.index("ytrack")], 
             'k-', linewidth=2, label='Reference Track')
    
    # Plot MPCC trajectory
    plt.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], 
             'r-', linewidth=2, label='MPCC Trajectory')
    plt.plot(ego_trajectory[0, 0], ego_trajectory[0, 1], 
             'go', markersize=8, label='Start')
    plt.plot(ego_trajectory[-1, 0], ego_trajectory[-1, 1], 
             'ro', markersize=8, label='End')
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('MPCC on RacetrackEnvSingle Track')
    plt.legend()
    plt.savefig('mpcc_rl_track_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return track_name, smax


if __name__ == "__main__":
    extract_and_run_mpcc()