#!/usr/bin/env python3
"""
MPCC wrapper that supports both pure MPCC and DRL-MPCC integration modes.

This wrapper handles:
1. Solver selection based on control mode
2. Parameter vector construction for DRL-enabled solver
3. Seamless switching between pure MPCC and integrated mode
"""

import numpy as np
import yaml
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MPCC.MPCC_class import MPCC
from integration.forces_pro_setting_with_drl import forces_pro_setting_with_drl
from simulator.dynamics import dynamics_simulator


class MPCCDRLWrapper:
    """
    MPCC wrapper supporting pure MPCC and DRL-MPCC integration.

    Modes:
    - pure_mpcc: Standard MPCC (17 parameters)
    - drl_mpcc: DRL-integrated MPCC (23 parameters with reference tracking)
    """

    def __init__(
        self,
        track,
        Tsim,
        vehicleparams,
        mpccparams,
        control_mode="pure_mpcc"
    ):
        """
        Initialize MPCC wrapper.

        Args:
            track: Track dictionary
            Tsim: Simulation time
            vehicleparams: Vehicle parameters dict
            mpccparams: MPCC parameters dict
            control_mode: "pure_mpcc" or "drl_mpcc"
        """
        self.control_mode = control_mode
        self.vehicleparams = vehicleparams
        self.mpccparams = mpccparams
        self.track = track

        # Core MPCC parameters
        self.Tf = mpccparams["Tf"]
        self.N = mpccparams["N"]
        self.Nsim = int(np.floor(self.N / self.Tf * Tsim))

        self.Qc = mpccparams["Qc"]
        self.Ql = mpccparams["Ql"]
        self.Q_theta = mpccparams["Q_theta"]
        self.R_a = mpccparams["R_a"]
        self.R_delta = mpccparams["R_delta"]

        self.r = track["r"]
        self.smax = track["smax"]
        self.track_lu_table = track["track_lu_table"]
        self.ppm = track.get("ppm", 100)

        # Track variables
        self.trackvars = [
            "sval", "tval", "xtrack", "ytrack", "phitrack",
            "cos(phi)", "sin(phi)", "g_upper", "g_lower",
        ]
        self.xvars = ["posx", "posy", "phi", "vx", "vy", "omega", "a", "delta", "theta"]
        self.uvars = ["jerk", "deltadot", "thetadot"]
        self.zvars = [
            "jerk", "deltadot", "thetadot",
            "posx", "posy", "phi", "vx", "vy", "omega", "a", "delta", "theta",
        ]

        # DRL integration parameters (for drl_mpcc mode)
        self.u_ref = (0.0, 0.0)  # (a_ref, delta_ref)
        self.omega_weights = None
        self.K_ref = np.array([100.0, 100.0])  # [K_a, K_delta]
        self.beta = 1.0

        # Parameter vector variables
        if control_mode == "drl_mpcc":
            self.pvars = [
                # Original 17 parameters
                "xt", "yt", "phit", "sin_phit", "cos_phit", "theta_hat",
                "Qc", "Ql", "Q_theta", "R_a", "R_delta", "r",
                "x_ob", "y_ob", "phi_ob", "l_ob", "w_ob",
                # DRL parameters (6 new)
                "a_ref", "delta_ref", "omega", "K_a", "K_delta", "beta",
            ]
        else:
            self.pvars = [
                "xt", "yt", "phit", "sin_phit", "cos_phit", "theta_hat",
                "Qc", "Ql", "Q_theta", "R_a", "R_delta", "r",
                "x_ob", "y_ob", "phi_ob", "l_ob", "w_ob",
            ]

        # Initialize solver
        self._initialize_solver()

        # State storage
        self.z_current = np.zeros((self.N, len(self.zvars)))
        self.theta_current = np.zeros((self.N,))
        self.zinit_vals = np.zeros((self.Nsim, len(self.zvars)))
        self.z_data = np.zeros((self.Nsim, self.N, len(self.zvars)))
        self.simidx = 0
        self.laps = 0
        self.laptimer = 0
        self.laptimes = []

        print(f"[MPCC] Initialized in {control_mode} mode")

    def _initialize_solver(self):
        """Initialize appropriate solver based on control mode."""
        generate_solver = self.mpccparams.get("generate_solver", False)

        if self.control_mode == "drl_mpcc":
            print("[MPCC] Using DRL-enabled solver")
            self.solver = forces_pro_setting_with_drl(
                self.mpccparams,
                self.vehicleparams,
                generate_solver
            )
        else:
            print("[MPCC] Using standard MPCC solver")
            from MPCC.forces_pro_setting import forces_pro_setting
            self.solver = forces_pro_setting(
                self.mpccparams,
                self.vehicleparams,
                generate_solver
            )

    def set_drl_parameters(self, u_ref, omega_weights, K_ref, beta):
        """
        Set DRL integration parameters.

        Args:
            u_ref: (a_ref, delta_ref) tuple
            omega_weights: Array of ω(t_k) for each stage
            K_ref: [K_a, K_delta] weights
            beta: Overall weight
        """
        if self.control_mode != "drl_mpcc":
            print("[WARNING] DRL parameters ignored in pure_mpcc mode")
            return

        self.u_ref = u_ref
        self.omega_weights = omega_weights
        self.K_ref = K_ref
        self.beta = beta

    def _build_parameter_vector(self, track_lin_point, stageidx, enemyinfo):
        """
        Build parameter vector for solver.

        Returns different sized vectors based on control mode:
        - pure_mpcc: 17 parameters
        - drl_mpcc: 23 parameters
        """
        # Obstacle info
        if enemyinfo is not None:
            x_ob = enemyinfo["x_ob"]
            y_ob = enemyinfo["y_ob"]
            phi_ob = enemyinfo["phi_ob"]
            l_ob = enemyinfo["l_ob"]
            w_ob = enemyinfo["w_ob"]
        else:
            x_ob = y_ob = phi_ob = l_ob = w_ob = 0.0

        # Base parameters (17)
        p_val = [
            track_lin_point[self.trackvars.index("xtrack")],
            track_lin_point[self.trackvars.index("ytrack")],
            track_lin_point[self.trackvars.index("phitrack")],
            track_lin_point[self.trackvars.index("sin(phi)")],
            track_lin_point[self.trackvars.index("cos(phi)")],
            track_lin_point[self.trackvars.index("sval")],
            self.Qc, self.Ql, self.Q_theta, self.R_a, self.R_delta, self.r,
            x_ob, y_ob, phi_ob, l_ob, w_ob,
        ]

        # Add DRL parameters if in drl_mpcc mode (6 more)
        if self.control_mode == "drl_mpcc":
            a_ref, delta_ref = self.u_ref
            omega = self.omega_weights[stageidx] if self.omega_weights is not None else 0.0
            K_a, K_delta = self.K_ref

            p_val.extend([
                a_ref, delta_ref,
                omega, K_a, K_delta,
                self.beta
            ])

        return np.array(p_val)

    def initialize_trajectory(self, xinit, enemyinfo, startidx):
        """Initialize MPCC trajectory."""
        # Initialize dynamics simulator
        self.dynamics = dynamics_simulator(
            self.vehicleparams, self.Tf / self.N, xinit, nodes=4
        )

        self.zinit = np.concatenate([np.array([0, 0, 0]), xinit])
        self.z_current = np.tile(self.zinit, (self.N, 1))

        # Initialize theta values
        theta_old = self.zinit[self.zvars.index("theta")] * np.ones((self.N,)) + 0.1 * np.arange(self.N)
        self.z_current[:, self.zvars.index("theta")] = theta_old

        index_lin_points = self.ppm * theta_old
        index_lin_points = np.clip(index_lin_points.astype(np.int32), 0, len(self.track_lu_table) - 1)
        track_lin_points = self.track_lu_table[index_lin_points, :]

        # Initialize positions on track
        self.z_current[:, 3] = track_lin_points[:, self.trackvars.index("xtrack")]
        self.z_current[:, 4] = track_lin_points[:, self.trackvars.index("ytrack")]
        self.z_current[:, 5] = track_lin_points[:, self.trackvars.index("phitrack")]

        # Iterative initialization
        for _ in range(80):
            all_parameters = []
            index_lin_points = self.ppm * theta_old
            index_lin_points = np.clip(index_lin_points.astype(np.int32), 0, len(self.track_lu_table) - 1)
            track_lin_points = self.track_lu_table[index_lin_points, :]

            for stageidx in range(self.N):
                p_val = self._build_parameter_vector(
                    track_lin_points[stageidx, :], stageidx, enemyinfo
                )
                all_parameters.append(p_val)

            all_parameters = np.array(all_parameters)

            problem = {
                "x0": self.z_current.reshape(-1,),
                "xinit": xinit,
                "all_parameters": all_parameters.reshape(-1,),
            }

            output, _, _ = self.solver.solve(problem)

            idx_sol = 0
            for key in output:
                self.z_current[idx_sol, :] = output[key]
                idx_sol += 1

            self.theta_current = self.z_current[:, self.zvars.index("theta")]
            theta_old = self.theta_current
            self.xinit = self.z_current[0, 3:]

        return self.z_current

    def update(self, enemyinfo):
        """
        Update MPCC controller.

        If in drl_mpcc mode, uses DRL reference parameters.
        If in pure_mpcc mode, ignores DRL parameters.
        """
        all_parameters = []

        theta_old = self.theta_current
        index_lin_points = self.ppm * theta_old
        index_lin_points = np.clip(index_lin_points.astype(np.int32), 0, len(self.track_lu_table) - 1)
        track_lin_points = self.track_lu_table[index_lin_points, :]

        # Build parameter vectors for all stages
        for stageidx in range(self.N):
            p_val = self._build_parameter_vector(
                track_lin_points[stageidx, :], stageidx, enemyinfo
            )
            all_parameters.append(p_val)

        all_parameters = np.array(all_parameters)

        # Solve optimization problem
        problem = {
            "x0": self.z_current.reshape(-1,),
            "xinit": self.xinit,
            "all_parameters": all_parameters.reshape(-1,),
        }

        output, _, _ = self.solver.solve(problem)

        # Extract solution
        idx_sol = 0
        for key in output:
            self.z_current[idx_sol, :] = output[key]
            idx_sol += 1

        # Simulate dynamics
        u = self.z_current[0, :3]
        self.dynamics.tick(u)
        self.dynamics.wrap_phi()
        xtrue = self.dynamics.x

        # Warmstart for next iteration
        self.z_current[1, 3:] = xtrue
        self.z_current = np.roll(self.z_current, -1, axis=0)
        self.z_current[-1, :] = self.z_current[-2, :]
        self.z_current[-1, self.zvars.index("theta")] += 0.1

        # Logging
        if self.simidx < self.Nsim:
            self.z_data[self.simidx, :, :] = self.z_current

        self.zinit = self.z_current[0, :]
        self.xinit = self.zinit[3:]

        if self.simidx < self.Nsim:
            self.zinit_vals[self.simidx, :] = self.zinit

        self.theta_current = self.z_current[:, self.zvars.index("theta")]

        # Handle lap completion
        if self.theta_current[0] > self.smax:
            self.laps += 1
            self.laptimes.append(self.simidx - self.laptimer)
            self.laptimer = self.simidx
            self.theta_current = self.theta_current - self.smax
            self.z_current[:, self.zvars.index("theta")] = self.theta_current
            self.dynamics.set_theta(self.theta_current[0])

        self.simidx += 1
        return self.z_current

    def return_sim_data(self):
        """Return simulation data."""
        return self.zinit_vals, self.z_data, np.array(self.laptimes)
