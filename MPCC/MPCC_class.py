import numpy as np
from MPCC.forces_pro_setting import forces_pro_setting

# from generate_tv_col_avoid_solver import get_tv_col_avoid_solver
import forcespro.nlp
from simulator.python_sim_utils import plotter, plot_pajecka, compute_objective
import matplotlib.pyplot as plt
import yaml
import sys
from simulator.dynamics import dynamics_simulator


class MPCC:

    def __init__(
        self,
        track,
        Tsim,
        vehicleparams,
        mpccparams,
    ):

        self.vehicleparams = vehicleparams

        self.Tf = mpccparams["Tf"]
        self.N = mpccparams["N"]
        self.Nsim = int(np.floor(self.N / self.Tf * Tsim))

        self.Qc = mpccparams["Qc"]
        self.Ql = mpccparams["Ql"]
        self.Q_theta = mpccparams["Q_theta"]
        self.R_a = mpccparams["R_a"]
        self.R_delta = mpccparams["R_delta"]
        self.generate_solver = mpccparams["generate_solver"]

        self.r = track["r"]
        self.smax = track["smax"]
        self.track_lu_table = track["track_lu_table"]
        self.ppm = track.get("ppm", 100)  # points per meter, default 100 for backward compatibility

        # DRL-MPCC Integration parameters (optional, set by SafetyAwareDRLMPCC)
        self.u_ref = None  # (a_ref, delta_ref) from DRL policy
        self.omega_weights = None  # Time-decaying weights ω(t_k)
        self.K_ref = None  # Reference tracking weights [K_a, K_delta]
        self.beta = 1.0  # Overall weight for reference tracking

        self.solver = forces_pro_setting(
            mpccparams, vehicleparams, self.generate_solver
        )
        # dynamics initialized in trajectory initialization

        self.trackvars = [
            "sval",
            "tval",
            "xtrack",
            "ytrack",
            "phitrack",
            "cos(phi)",
            "sin(phi)",
            "g_upper",
            "g_lower",
        ]
        self.xvars = ["posx", "posy", "phi", "vx", "vy", "omega", "a", "delta", "theta"]
        self.uvars = ["jerk", "deltadot", "thetadot"]
        self.pvars = [
            "xt",
            "yt",
            "phit",
            "sin_phit",
            "cos_phit",
            "theta_hat",
            "Qc",
            "Ql",
            "Q_theta",
            "R_a",
            "R_delta",
            "r",
            "x_ob",
            "y_ob",
            "phi_ob",
            "l_ob",
            "w_ob",
        ]
        self.zvars = [
            "jerk",
            "deltadot",
            "thetadot",
            "posx",
            "posy",
            "phi",
            "vx",
            "vy",
            "omega",
            "a",
            "delta",
            "theta",
        ]

        self.z_current = np.zeros((self.N, len(self.zvars)))
        self.theta_current = np.zeros((self.N,))

        # list to store all visited states
        self.zinit_vals = np.zeros((self.Nsim, len(self.zvars)))
        # list containing also the prediction horizons
        self.z_data = np.zeros((self.Nsim, self.N, len(self.zvars)))
        # sim step trackign
        self.simidx = 0
        self.laps = 0
        self.laptimer = 0
        self.laptimes = []

    def initialize_trajectory(self, xinit, enemyinfo, startidx):
        if enemyinfo is not None:
            x_ob = enemyinfo["x_ob"]
            y_ob = enemyinfo["y_ob"]
            phi_ob = enemyinfo["phi_ob"]
            l_ob = enemyinfo["l_ob"]
            w_ob = enemyinfo["w_ob"]
        else:
            # No obstacle
            x_ob = 0
            y_ob = 0
            phi_ob = 0
            l_ob = 0
            w_ob = 0

        # initialization for theta values
        iter = 80

        # initialize dyamics simulation
        self.dynamics = dynamics_simulator(
            self.vehicleparams, self.Tf / self.N, xinit, nodes=4
        )

        self.zinit = np.concatenate([np.array([0, 0, 0]), xinit])
        self.z_current = np.tile(self.zinit, (self.N, 1))

        print(f"Initial xinit: {xinit}")
        print(f"Initial zinit: {self.zinit}")
        print(f"smax: {self.smax}, ppm: {self.ppm}")

        # arbitrarily set theta  values and
        theta_old = self.zinit[self.zvars.index("theta")] * np.ones(
            (self.N,)
        ) + 0.1 * np.arange(self.N)
        self.z_current[:, self.zvars.index("theta")] = theta_old
        print(f"Initial theta values: {theta_old[:5]}...{theta_old[-5:]}")
        index_lin_points = self.ppm * theta_old
        index_lin_points = np.clip(index_lin_points.astype(np.int32), 0, len(self.track_lu_table) - 1)
        track_lin_points = self.track_lu_table[index_lin_points, :]

        # initialize x values on track
        self.z_current[:, 3] = track_lin_points[:, self.trackvars.index("xtrack")]
        self.z_current[:, 4] = track_lin_points[:, self.trackvars.index("ytrack")]
        self.z_current[:, 5] = track_lin_points[:, self.trackvars.index("phitrack")]

        for idx in range(iter):
            # print("theta values", theta_old)
            all_parameters = []
            # get track linearization
            index_lin_points = self.ppm * theta_old
            index_lin_points = np.clip(index_lin_points.astype(np.int32), 0, len(self.track_lu_table) - 1)
            track_lin_points = self.track_lu_table[index_lin_points, :]

            for stageidx in range(self.N):
                p_val = np.array(
                    [
                        track_lin_points[stageidx, self.trackvars.index("xtrack")],
                        track_lin_points[stageidx, self.trackvars.index("ytrack")],
                        track_lin_points[stageidx, self.trackvars.index("phitrack")],
                        track_lin_points[stageidx, self.trackvars.index("sin(phi)")],
                        track_lin_points[stageidx, self.trackvars.index("cos(phi)")],
                        track_lin_points[
                            stageidx, self.trackvars.index("sval")
                        ],  # aka theta_hat
                        self.Qc,
                        self.Ql,
                        self.Q_theta,
                        self.R_a,
                        self.R_delta,
                        self.r,
                        x_ob,
                        y_ob,
                        phi_ob,
                        l_ob,
                        w_ob,
                    ]
                )
                all_parameters.append(p_val)

            all_parameters = np.array(all_parameters)

            # problem dictionary, arrays have to be flattened
            problem = {
                "x0": self.z_current.reshape(
                    -1,
                ),
                "xinit": xinit,
                "all_parameters": all_parameters.reshape(
                    -1,
                ),
            }
            # solve problem
            output, exitflag, info = self.solver.solve(problem)
            print("Initialization - info:", info)
            print("Initialization - exitflag:", exitflag)

            # extract theta values
            idx_sol = 0
            for key in output:
                # print(key)
                zsol = output[key]
                usol = zsol[0:3]
                self.z_current[idx_sol, :] = zsol
                idx_sol = idx_sol + 1

            self.theta_current = self.z_current[:, self.zvars.index("theta")]

            # compute difference
            theta_diff = np.sum(np.abs(self.theta_current - theta_old))
            # print("theta init difference: ", theta_diff)
            # print("theta values", self.theta_current)
            theta_old = self.theta_current
            self.xinit = self.z_current[0, 3:]
        return self.z_current

    def update(self, enemyinfo):

        if enemyinfo is not None:
            x_ob = enemyinfo["x_ob"]
            y_ob = enemyinfo["y_ob"]
            phi_ob = enemyinfo["phi_ob"]
            l_ob = enemyinfo["l_ob"]
            w_ob = enemyinfo["w_ob"]
        else:
            # No obstacle
            x_ob = 0
            y_ob = 0
            phi_ob = 0
            l_ob = 0
            w_ob = 0

        all_parameters = []

        theta_old = self.theta_current
        # get track linearization
        index_lin_points = self.ppm * theta_old
        index_lin_points = np.clip(index_lin_points.astype(np.int32), 0, len(self.track_lu_table) - 1)
        # print("track linearized around entries:", index_lin_points)
        track_lin_points = self.track_lu_table[index_lin_points, :]

        #######################################################################
        # set params and warmstart
        for stageidx in range(self.N - 1):
            p_val = np.array(
                [
                    track_lin_points[stageidx, self.trackvars.index("xtrack")],
                    track_lin_points[stageidx, self.trackvars.index("ytrack")],
                    track_lin_points[stageidx, self.trackvars.index("phitrack")],
                    track_lin_points[stageidx, self.trackvars.index("sin(phi)")],
                    track_lin_points[stageidx, self.trackvars.index("cos(phi)")],
                    track_lin_points[
                        stageidx, self.trackvars.index("sval")
                    ],  # aka theta_hat
                    self.Qc,
                    self.Ql,
                    self.Q_theta,
                    self.R_a,
                    self.R_delta,
                    self.r,
                    x_ob,
                    y_ob,
                    phi_ob,
                    l_ob,
                    w_ob,
                ]
            )
            # create parameter matrix
            all_parameters.append(p_val)

        # last stage copy old solution for init
        stageidx = self.N - 1
        p_val = np.array(
            [
                track_lin_points[stageidx, self.trackvars.index("xtrack")],
                track_lin_points[stageidx, self.trackvars.index("ytrack")],
                track_lin_points[stageidx, self.trackvars.index("phitrack")],
                track_lin_points[stageidx, self.trackvars.index("sin(phi)")],
                track_lin_points[stageidx, self.trackvars.index("cos(phi)")],
                track_lin_points[
                    stageidx, self.trackvars.index("sval")
                ],  # aka theta_hat
                self.Qc,
                self.Ql,
                self.Q_theta,
                self.R_a,
                self.R_delta,
                self.r,
                x_ob,
                y_ob,
                phi_ob,
                l_ob,
                w_ob,
            ]
        )
        all_parameters.append(p_val)
        all_parameters = np.array(all_parameters)
        # last state of z_current is already copied.

        #######################################################################
        # problem dictionary, arrays have to be flattened
        problem = {
            "x0": self.z_current.reshape(
                -1,
            ),
            "xinit": self.xinit,
            "all_parameters": all_parameters.reshape(
                -1,
            ),
        }
        # solve problem
        output, exitflag, info = self.solver.solve(problem)
        # print("exitflag = ", exitflag)
        # print("xinit ", self.xinit)
        # extract solution
        idx_sol = 0
        for key in output:
            # print(key)
            zsol = output[key]
            self.z_current[idx_sol, :] = zsol
            idx_sol = idx_sol + 1

        # simulate dynaics
        u = self.z_current[0, :3]
        xtrue = self.dynamics.tick(u)  # self.z_current[1, 3:] #
        
        # wrap phi angle to keep it in [0, 2π] range
        self.dynamics.wrap_phi()
        xtrue = self.dynamics.x  # get updated state with wrapped phi

        # shift horizon for next warmstart and instert the new "measured position"
        self.z_current[1, 3:] = xtrue
        self.z_current = np.roll(self.z_current, -1, axis=0)
        self.z_current[-1, :] = self.z_current[-2, :]
        # advance the last prediction for theta
        self.z_current[-1, self.zvars.index("theta")] += 0.1

        # log solution (only if within bounds)
        if self.simidx < self.Nsim:
            self.z_data[self.simidx, :, :] = self.z_current
        self.zinit = self.z_current[0, :]
        self.xinit = self.zinit[3:]
        if self.simidx < self.Nsim:
            self.zinit_vals[self.simidx, :] = self.zinit

        self.theta_current = self.z_current[:, self.zvars.index("theta")]

        if self.theta_current[0] > self.smax:
            print(
                "#################################RESET###############################"
            )
            self.laps = self.laps + 1
            self.laptimes.append(self.simidx - self.laptimer)
            self.laptimer = self.simidx
            print("lap:", self.laps)
            self.theta_current = self.theta_current - self.smax
            self.z_current[:, self.zvars.index("theta")] = self.theta_current
            self.dynamics.set_theta(self.theta_current[0])

        self.simidx = self.simidx + 1
        return self.z_current

    def return_sim_data(self):
        return self.zinit_vals, self.z_data, np.array(self.laptimes)
