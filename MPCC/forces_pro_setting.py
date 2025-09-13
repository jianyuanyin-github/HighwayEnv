import casadi
import yaml
import numpy as np
import forcespro.nlp


def forces_pro_setting(
    mpccparams,
    vehicleparams,
    generate_solver=True,
):
    if not generate_solver:
        print("[SOLVERINFO] Loading solver from directory")
        solver = forcespro.nlp.Solver.from_directory("./dynamics_solver_forcespro")
        return solver

    # with open(vehicleparams) as file:
    #     mpccparams = yaml.load(file, Loader=yaml.FullLoader)

    m = vehicleparams["m"]  # [kg]
    lf = vehicleparams["lf"]  # [m]
    lr = vehicleparams["lr"]  # [m]
    Iz = vehicleparams["Iz"]  # [kg*m^3]
    lencar = vehicleparams["veh_length"]
    widthcar = vehicleparams["veh_width"]

    # pajecka and motor coefficients
    Bf = vehicleparams["Bf"]
    Br = vehicleparams["Br"]
    Cf = vehicleparams["Cf"]
    Cr = vehicleparams["Cr"]
    Df = vehicleparams["Df"]
    Dr = vehicleparams["Dr"]

    ro = vehicleparams["ro"]
    s = vehicleparams["s"]
    Cd = vehicleparams["Cd"]
    fr0 = vehicleparams["fr0"]
    fr1 = vehicleparams["fr1"]
    fr4 = vehicleparams["fr4"]

    # load solverparams for the boxconstraints
    # with open(mpccparams) as file:
    #     mpccparams = yaml.load(file, Loader=yaml.FullLoader)

    N = mpccparams["N"]  # horizon steps
    Tf = mpccparams["Tf"]  # horizon length [s]

    jerk_min = mpccparams["jerk_min"]  # min change in a [m/s^3]
    jerk_max = mpccparams["jerk_max"]  # max change in a [m/s^3]

    a_min = mpccparams["a_min"]  # min a [m/s^2]
    a_max = mpccparams["a_max"]  # max a [m/s^2]

    delta_min = mpccparams["delta_min"]  # minimum steering angle [rad]
    delta_max = mpccparams["delta_max"]  # maximum steering angle [rad]

    deltadot_min = mpccparams["deltadot_min"]  # minimum steering angle cahgne[rad/s]
    deltadot_max = mpccparams["deltadot_max"]  # maximum steering angle cahgne[rad/s]

    omega_min = mpccparams["omega_min"]  # minimum yawrate [rad/sec]
    omega_max = mpccparams["omega_max"]  # maximum yawrate [rad/sec]

    thetadot_min = mpccparams["thetadot_min"]  # minimum adv param speed [m/s]
    thetadot_max = mpccparams["thetadot_max"]  # maximum adv param speed [m/s]

    theta_min = mpccparams["theta_min"]  # minimum adv param [m]
    theta_max = mpccparams["theta_max"]  # maximum adv param  [m]
    
    phi_min = mpccparams.get("phi_min", -6.28)  # minimum heading angle [rad]
    phi_max = mpccparams.get("phi_max", 6.28)   # maximum heading angle [rad]

    vx_min = mpccparams["vx_min"]  # min long vel [m/s]
    vx_max = mpccparams["vx_max"]  # max long vel [m/s]

    vy_min = mpccparams["vy_min"]  # min lat vel [m/s]
    vy_max = mpccparams["vy_max"]  # max lat vel [m/s]

    # forces model
    model = forcespro.nlp.SymbolicModel()

    # compute sampling time for integration of continuous dynamics
    Ts = Tf / N

    # set dimensions
    model.N = N
    model.nvar = 12  # stage variables z = [u, x]'
    model.neq = 9  # number of equality constraints
    model.nh = 2  # number of inequality constraints
    model.npar = 17  #
    ninputs = 3

    # let z = [u, x] = [jerk, deltadot, thetadot, posx, posy, phi, vx, vy, omega, a, delta, theta]
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
    pvars = [
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

    # define objective
    def stage_cost(z, p):
        # extract parameters
        xt = p[pvars.index("xt")]
        yt = p[pvars.index("yt")]
        phit = p[pvars.index("phit")]
        sin_phit = p[pvars.index("sin_phit")]
        cos_phit = p[pvars.index("cos_phit")]
        theta_hat = p[pvars.index("theta_hat")]
        Qc = p[pvars.index("Qc")]
        Ql = p[pvars.index("Ql")]
        Q_theta = p[pvars.index("Q_theta")]
        R_a = p[pvars.index("R_a")]
        R_delta = p[pvars.index("R_delta")]

        # extract states
        posx = z[zvars.index("posx")]
        posy = z[zvars.index("posy")]
        theta = z[zvars.index("theta")]

        # extract inputs
        jerk = z[zvars.index("jerk")]
        deltadot = z[zvars.index("deltadot")]
        thetadot = z[zvars.index("thetadot")]

        # compute approximate linearized contouring and lag error
        xt_hat = xt + cos_phit * (theta - theta_hat)
        yt_hat = yt + sin_phit * (theta - theta_hat)

        e_cont = sin_phit * (xt_hat - posx) - cos_phit * (yt_hat - posy)
        e_lag = cos_phit * (xt_hat - posx) + sin_phit * (yt_hat - posy)

        cost = (
            e_cont * Qc * e_cont
            + e_lag * Ql * e_lag
            - Q_theta * thetadot
            + jerk * R_a * jerk
            + deltadot * R_delta * deltadot
        )

        return cost

    model.objective = lambda z, p: stage_cost(z, p)

    def continuous_dynamics(x, u, p):
        # extract states and inputs
        posx = x[zvars.index("posx") - ninputs]
        posy = x[zvars.index("posy") - ninputs]
        phi = x[zvars.index("phi") - ninputs]
        vx = x[zvars.index("vx") - ninputs]
        vy = x[zvars.index("vy") - ninputs]
        omega = x[zvars.index("omega") - ninputs]
        a = x[zvars.index("a") - ninputs]
        delta = x[zvars.index("delta") - ninputs]
        theta = x[zvars.index("theta") - ninputs]

        jerk = u[zvars.index("jerk")]
        deltadot = u[zvars.index("deltadot")]
        thetadot = u[zvars.index("thetadot")]

        # build CasADi expressions for dynamic model
        # front lateral tireforce
        alphaf = -casadi.atan2((omega * lf + vy), vx) + delta
        Ffy = Df * casadi.sin(Cf * casadi.atan(Bf * alphaf))

        # rear lateral tireforce
        alphar = casadi.atan2((omega * lr - vy), vx)
        Fry = Dr * casadi.sin(Cr * casadi.atan(Br * alphar))

        # rear longitudinal forces
        Faero = 0.5 * ro * s * Cd * vx * vx
        Froll = fr0 + fr1 * vx / 100 + fr4 * (vx / 100**4)
        Frx = m * a - Froll - Faero

        # let z = [u, x] = [jerk, deltadot, thetadot, posx, posy, phi, vx, vy, omega, a, delta, theta]

        statedot = casadi.vertcat(
            vx * casadi.cos(phi) - vy * casadi.sin(phi),  # posxdot
            vx * casadi.sin(phi) + vy * casadi.cos(phi),  # posydot
            omega,  # phidot
            1 / m * (Frx - Ffy * casadi.sin(delta) + m * vy * omega),  # vxdot
            1 / m * (Fry + Ffy * casadi.cos(delta) - m * vx * omega),  # vydot
            1 / Iz * (Ffy * lf * casadi.cos(delta) - Fry * lr),  # omegadot
            jerk,  # adot = jerk
            deltadot,
            thetadot,
        )
        return statedot

    # set model to continuous dynamics mode
    model.continuous_dynamics = continuous_dynamics

    # dynamics only in state Variables
    model.E = np.concatenate([np.zeros((9, 3)), np.eye(9)], axis=1)

    # nonlinear constraints
    def nonlinear_ineq(z, p):
        # extract parameters
        xt = p[pvars.index("xt")]
        yt = p[pvars.index("yt")]
        phit = p[pvars.index("phit")]
        sin_phit = p[pvars.index("sin_phit")]
        cos_phit = p[pvars.index("cos_phit")]
        theta_hat = p[pvars.index("theta_hat")]
        r = p[pvars.index("r")]

        # extract relevant states
        posx = z[zvars.index("posx")]
        posy = z[zvars.index("posy")]
        theta = z[zvars.index("theta")]

        # compute approximate linearized contouring and lag error
        xt_hat = xt + cos_phit * (theta - theta_hat)
        yt_hat = yt + sin_phit * (theta - theta_hat)

        # inside track <=> tval <= 0
        tval = (xt_hat - posx) ** 2 + (yt_hat - posy) ** 2 - (r - widthcar) ** 2

        # ellipsoidal obstacle

        x_ob = p[pvars.index("x_ob")]
        y_ob = p[pvars.index("y_ob")]
        phi_ob = p[pvars.index("phi_ob")]
        l_ob = p[pvars.index("l_ob")]
        w_ob = p[pvars.index("w_ob")]

        # implicit elipse eqn
        dx = posx - x_ob
        dy = posy - y_ob
        s = casadi.sin(phi_ob)
        c = casadi.cos(phi_ob)
        # tighten constraint with car length/width
        a = np.sqrt(2) * (l_ob / 2 + lencar / 2)
        b = np.sqrt(2) * (w_ob / 2 + widthcar / 2)
        # implicit ellipse value ielval = 1 defines obstacle ellipse
        ielval = (1 / a**2) * (c * dx + s * dy) * (c * dx + s * dy) + (1 / b**2) * (
            s * dx - c * dy
        ) * (s * dx - c * dy)
        # cosntraint value -> obsval<=0  <=> car outside of obstacle
        obsval = 1 / (1 + casadi.exp(-(ielval - 1)))

        # concatenate
        hval = casadi.vertcat(tval, obsval)
        return hval

    model.ineq = lambda z, p: nonlinear_ineq(z, p)
    model.hu = np.array([0.0000, 2])
    model.hl = np.array([-10, 0.51])

    # boxconstraints
    # Note: z = [u, x] = [jerk, deltadot, thetadot, posx, posy, phi, vx, vy, omega, a, delta, theta]
    model.ub = np.array(
        [
            jerk_max,
            deltadot_max,
            thetadot_max,
            1e6,  # posx_max - remove constraint
            1e6,  # posy_max - remove constraint
            1e6,  # phi_max - remove constraint (use wrapping instead)
            vx_max,
            vy_max,
            omega_max,
            a_max,
            delta_max,
            theta_max,
        ]
    )
    model.lb = np.array(
        [
            jerk_min,
            deltadot_min,
            thetadot_min,
            -1e6,  # posx_min - remove constraint
            -1e6,  # posy_min - remove constraint
            -1e6,  # phi_min - remove constraint (use wrapping instead)
            vx_min,
            vy_min,
            omega_min,
            a_min,
            delta_min,
            theta_min,
        ]
    )

    # put initial condition on all state variables x
    model.xinitidx = 3 + np.arange(model.nvar - 3)
    # Set solver options
    codeoptions = forcespro.CodeOptions("dynamics_solver_forcespro")
    codeoptions.nlp.integrator.type = "ERK4"
    codeoptions.nlp.integrator.Ts = Ts
    codeoptions.nlp.integrator.nodes = 2  # intermediate integration nodes

    codeoptions.maxit = 30  # Maximum number of iterations
    codeoptions.printlevel = (
        0
        # Use printlevel = 2 to print progress (but not for timings)
    )
    codeoptions.optlevel = 2  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
    codeoptions.nlp.stack_parambounds = 2
    # codeoptions.noVariableElimination = True
    # Creates code for symbolic model formulation given above, then contacts server to generate new solver
    solver = model.generate_solver(codeoptions)
    return solver
