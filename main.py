import numpy as np
from MPCC.MPCC_class import MPCC
import forcespro.nlp
from simulator.python_sim_utils import plotter, plot_pajecka, compute_objective
import matplotlib.pyplot as plt
import tracks.InterpolateTrack as InterpolateTrack
import yaml
import sys
import pickle
from datetime import datetime
from pathlib import Path


def main():

    # model parameters
    vehicleparams_file = "config/vehicleparams.yaml"
    mpccparams_file = "config/mpccparams.yaml"
    simulatorparams_file = "config/simulationparams.yaml"

    # load global constant model parameters
    with open(vehicleparams_file) as file:
        vehicleparams = yaml.load(file, Loader=yaml.FullLoader)
    lf = vehicleparams["lf"]  # [m]
    lr = vehicleparams["lr"]  # [m]
    l = lf + lr

    # sim parameters
    with open(mpccparams_file) as file:
        mpccparams = yaml.load(file, Loader=yaml.FullLoader)
    Tsim = mpccparams["Tsim"]
    Tf = mpccparams["Tf"]
    N = mpccparams["N"]
    Nsim = int(np.floor(N * Tsim / Tf))  # number of simulation steps

    with open(simulatorparams_file) as file:
        simulationparams = yaml.load(file, Loader=yaml.FullLoader)
    trackname = simulationparams["trackname"]
    r = simulationparams["r"]  # trackwidth
    track_lu_table, smax = InterpolateTrack.generatelookuptable(
        "tracks/" + trackname, r
    )

    # Load track parameters to get ppm
    with open("tracks/" + trackname + "_params.yaml") as file:
        track_params = yaml.load(file, Loader=yaml.FullLoader)
    ppm = track_params["ppm"]

    track = {"track_lu_table": track_lu_table, "smax": smax, "r": r, "ppm": ppm}

    lencar = vehicleparams["veh_length"]  # Vehicle length from config
    trk_plt = plotter(
        track_lu_table,
        smax,
        r,
        lencar,
        ppm,
    )

    MPCC_Controller = MPCC(track, Tsim, vehicleparams, mpccparams)

    # plot_pajecka(paramfile)
    trk_plt.plot_track()

    # starting position in track startidx = theta0[m] * 100 [pts/m]
    trackvars = [
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
    startidx = simulationparams["startidx"]
    xt0 = track_lu_table[startidx, trackvars.index("xtrack")]
    yt0 = track_lu_table[startidx, trackvars.index("ytrack")]
    phit0 = track_lu_table[startidx, trackvars.index("phitrack")]
    theta_hat0 = track_lu_table[startidx, trackvars.index("sval")]
    # initial condition
    zvars = [
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
    vx0 = simulationparams["vx0"]
    xvars = ["posx", "posy", "phi", "vx", "vy", "omega", "a", "delta", "theta"]
    xinit = np.array([xt0, yt0, phit0, vx0, 0.0, 0, 0, 0, theta_hat0])
    # ---- [0, 2π]  ----
    xinit[2] = np.mod(xinit[2], 2 * np.pi)

    # static obstacle - temporarily disabled
    # ob_idx = simulationparams["ob_idx"]
    # phi_ob = track_lu_table[ob_idx, trackvars.index("phitrack")]
    # x_ob = track_lu_table[ob_idx, trackvars.index("xtrack")] + 1.5 * r * np.sin(phi_ob)
    # y_ob = track_lu_table[ob_idx, trackvars.index("ytrack")] - 1.5 * r * np.cos(phi_ob)
    # l_ob = 1.5 * lencar
    # w_ob = 1.5 * lencar
    obstacleinfo = None

    # trk_plt.plot_static_obstacle(x_ob, y_ob, phi_ob, l_ob, w_ob, color="red")
    z_current = MPCC_Controller.initialize_trajectory(xinit, obstacleinfo, startidx)
    trk_plt.plot_horizon(z_current[:, zvars.index("theta")], z_current[:, 3:6])
    plt.pause(0.1)
    trk_plt.clear_horizion()
    trk_plt.clear_input_state_traj()
    input("start")
    ##########################SIMULATION#######################################
    for simidx in range(Nsim):
        z_current = MPCC_Controller.update(obstacleinfo)
        #'''
        # plotting result
        trk_plt.plot_horizon(z_current[:, zvars.index("theta")], z_current[:, 3:6])
        trk_plt.plot_input_state_traj(z_current, zvars)

        plt.pause(0.01)
        # input("hit [enter] to continue.")
        # plt.pause(0.1)
        trk_plt.clear_horizion()
        trk_plt.clear_input_state_traj()
        #'''
    ###############################/SIMULATION##################################

    # zinit_vals, z_data, laptimes = MPCC_Controller.return_sim_data()

    return 0


if __name__ == "__main__":

    main()
