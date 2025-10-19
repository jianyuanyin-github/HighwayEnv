import casadi
import yaml
import numpy as np
import forcespro.nlp


def forces_pro_setting_with_drl(
    mpccparams,
    vehicleparams,
    generate_solver=True,
):
    """
    ForcesProforces solver with DRL reference tracking support.

    This version extends the parameter vector to include:
    - a_ref, delta_ref: Reference controls from DRL policy
    - omega: Time-decaying weight for reference tracking
    - K_a, K_delta: Reference tracking weights
    - beta: Overall reference tracking weight

    Cost function becomes:
    J = β·ω(t)·J_ref + (1-ω(t))·J_t

    where:
    - J_ref = (a - a_ref)^T·K·(a - a_ref)
    - J_t = standard MPCC cost (contouring, lag, progress, smoothness)
    """
    if not generate_solver:
        print("[SOLVERINFO] Loading DRL-enabled solver from directory")
        solver = forcespro.nlp.Solver.from_directory("./dynamics_solver_forcespro_drl")
        return solver

    # Vehicle parameters
    m = vehicleparams["m"]
    lf = vehicleparams["lf"]
    lr = vehicleparams["lr"]
    Iz = vehicleparams["Iz"]
    lencar = vehicleparams["veh_length"]
    widthcar = vehicleparams["veh_width"]

    # Tire model parameters
    Bf = vehicleparams["Bf"]
    Br = vehicleparams["Br"]
    Cf = vehicleparams["Cf"]
    Cr = vehicleparams["Cr"]
    Df = vehicleparams["Df"]
    Dr = vehicleparams["Dr"]

    # Drag parameters
    ro = vehicleparams["ro"]
    s = vehicleparams["s"]
    Cd = vehicleparams["Cd"]
    fr0 = vehicleparams["fr0"]
    fr1 = vehicleparams["fr1"]
    fr4 = vehicleparams["fr4"]

    # MPC parameters
    N = mpccparams["N"]
    Tf = mpccparams["Tf"]

    # Box constraints
    jerk_min = mpccparams["jerk_min"]
    jerk_max = mpccparams["jerk_max"]
    a_min = mpccparams["a_min"]
    a_max = mpccparams["a_max"]
    delta_min = mpccparams["delta_min"]
    delta_max = mpccparams["delta_max"]
    deltadot_min = mpccparams["deltadot_min"]
    deltadot_max = mpccparams["deltadot_max"]
    omega_min = mpccparams["omega_min"]
    omega_max = mpccparams["omega_max"]
    thetadot_min = mpccparams["thetadot_min"]
    thetadot_max = mpccparams["thetadot_max"]
    theta_min = mpccparams["theta_min"]
    theta_max = mpccparams["theta_max"]
    phi_min = mpccparams.get("phi_min", -6.28)
    phi_max = mpccparams.get("phi_max", 6.28)
    vx_min = mpccparams["vx_min"]
    vx_max = mpccparams["vx_max"]
    vy_min = mpccparams["vy_min"]
    vy_max = mpccparams["vy_max"]

    # ForcesProforces model
    model = forcespro.nlp.SymbolicModel()
    Ts = Tf / N

    # Model dimensions
    model.N = N
    model.nvar = 12  # z = [jerk, deltadot, thetadot, posx, posy, phi, vx, vy, omega, a, delta, theta]
    model.neq = 9
    model.nh = 2
    model.npar = 23  # EXTENDED: Added 6 parameters for DRL integration
    ninputs = 3

    # Decision variables
    zvars = [
        "jerk", "deltadot", "thetadot",
        "posx", "posy", "phi", "vx", "vy", "omega", "a", "delta", "theta",
    ]

    # Parameters (EXTENDED for DRL)
    pvars = [
        # Original MPCC parameters
        "xt", "yt", "phit", "sin_phit", "cos_phit", "theta_hat",
        "Qc", "Ql", "Q_theta", "R_a", "R_delta", "r",
        "x_ob", "y_ob", "phi_ob", "l_ob", "w_ob",
        # NEW: DRL reference tracking parameters
        "a_ref", "delta_ref",  # Reference controls from DRL
        "omega", "K_a", "K_delta",  # Weights
        "beta",  # Overall reference weight
    ]

    # Define INTEGRATED objective function
    def stage_cost(z, p):
        # Extract original MPCC parameters
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

        # Extract DRL reference tracking parameters
        a_ref = p[pvars.index("a_ref")]
        delta_ref = p[pvars.index("delta_ref")]
        omega = p[pvars.index("omega")]
        K_a = p[pvars.index("K_a")]
        K_delta = p[pvars.index("K_delta")]
        beta = p[pvars.index("beta")]

        # Extract states
        posx = z[zvars.index("posx")]
        posy = z[zvars.index("posy")]
        theta = z[zvars.index("theta")]
        a = z[zvars.index("a")]
        delta = z[zvars.index("delta")]

        # Extract control rates
        jerk = z[zvars.index("jerk")]
        deltadot = z[zvars.index("deltadot")]
        thetadot = z[zvars.index("thetadot")]

        # ===== Standard MPCC cost (J_t) =====
        # Contouring and lag errors
        xt_hat = xt + cos_phit * (theta - theta_hat)
        yt_hat = yt + sin_phit * (theta - theta_hat)
        e_cont = sin_phit * (xt_hat - posx) - cos_phit * (yt_hat - posy)
        e_lag = cos_phit * (xt_hat - posx) + sin_phit * (yt_hat - posy)

        J_t = (
            e_cont * Qc * e_cont +
            e_lag * Ql * e_lag -
            Q_theta * thetadot +
            jerk * R_a * jerk +
            deltadot * R_delta * deltadot
        )

        # ===== DRL reference tracking cost (J_ref) =====
        # Eq. (3) from paper: ||u - u_ref||_K^2
        error_a = a - a_ref
        error_delta = delta - delta_ref

        J_ref = K_a * error_a * error_a + K_delta * error_delta * error_delta

        # ===== Combined cost with time-decaying weight =====
        # Eq. (2) from paper: J = β·ω(t)·J_ref + (1-ω(t))·J_t
        cost = beta * omega * J_ref + (1.0 - omega) * J_t

        return cost

    model.objective = lambda z, p: stage_cost(z, p)

    # Dynamics (unchanged)
    def continuous_dynamics(x, u, p):
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

        # Tire forces
        alphaf = -casadi.atan2((omega * lf + vy), vx) + delta
        Ffy = Df * casadi.sin(Cf * casadi.atan(Bf * alphaf))

        alphar = casadi.atan2((omega * lr - vy), vx)
        Fry = Dr * casadi.sin(Cr * casadi.atan(Br * alphar))

        # Longitudinal forces
        Faero = 0.5 * ro * s * Cd * vx * vx
        Froll = fr0 + fr1 * vx / 100 + fr4 * (vx / 100**4)
        Frx = m * a - Froll - Faero

        statedot = casadi.vertcat(
            vx * casadi.cos(phi) - vy * casadi.sin(phi),
            vx * casadi.sin(phi) + vy * casadi.cos(phi),
            omega,
            1 / m * (Frx - Ffy * casadi.sin(delta) + m * vy * omega),
            1 / m * (Fry + Ffy * casadi.cos(delta) - m * vx * omega),
            1 / Iz * (Ffy * lf * casadi.cos(delta) - Fry * lr),
            jerk,
            deltadot,
            thetadot,
        )
        return statedot

    model.continuous_dynamics = continuous_dynamics
    model.E = np.concatenate([np.zeros((9, 3)), np.eye(9)], axis=1)

    # Constraints (unchanged)
    def nonlinear_ineq(z, p):
        xt = p[pvars.index("xt")]
        yt = p[pvars.index("yt")]
        phit = p[pvars.index("phit")]
        sin_phit = p[pvars.index("sin_phit")]
        cos_phit = p[pvars.index("cos_phit")]
        theta_hat = p[pvars.index("theta_hat")]
        r = p[pvars.index("r")]

        posx = z[zvars.index("posx")]
        posy = z[zvars.index("posy")]
        theta = z[zvars.index("theta")]

        xt_hat = xt + cos_phit * (theta - theta_hat)
        yt_hat = yt + sin_phit * (theta - theta_hat)

        # Track boundary
        tval = (xt_hat - posx) ** 2 + (yt_hat - posy) ** 2 - (r - widthcar) ** 2

        # Obstacle avoidance
        x_ob = p[pvars.index("x_ob")]
        y_ob = p[pvars.index("y_ob")]
        phi_ob = p[pvars.index("phi_ob")]
        l_ob = p[pvars.index("l_ob")]
        w_ob = p[pvars.index("w_ob")]

        dx = posx - x_ob
        dy = posy - y_ob
        s = casadi.sin(phi_ob)
        c = casadi.cos(phi_ob)
        a = np.sqrt(2) * (l_ob / 2 + lencar / 2)
        b = np.sqrt(2) * (w_ob / 2 + widthcar / 2)
        ielval = (1 / a**2) * (c * dx + s * dy) * (c * dx + s * dy) + (1 / b**2) * (s * dx - c * dy) * (s * dx - c * dy)
        obsval = 1 / (1 + casadi.exp(-(ielval - 1)))

        hval = casadi.vertcat(tval, obsval)
        return hval

    model.ineq = lambda z, p: nonlinear_ineq(z, p)
    model.hu = np.array([0.0000, 2])
    model.hl = np.array([-10, 0.51])

    # Box constraints
    model.ub = np.array([
        jerk_max, deltadot_max, thetadot_max,
        1e6, 1e6, 1e6,
        vx_max, vy_max, omega_max,
        a_max, delta_max, theta_max,
    ])
    model.lb = np.array([
        jerk_min, deltadot_min, thetadot_min,
        -1e6, -1e6, -1e6,
        vx_min, vy_min, omega_min,
        a_min, delta_min, theta_min,
    ])

    model.xinitidx = 3 + np.arange(model.nvar - 3)

    # Solver options
    codeoptions = forcespro.CodeOptions("dynamics_solver_forcespro_drl")
    codeoptions.nlp.integrator.type = "ERK4"
    codeoptions.nlp.integrator.Ts = Ts
    codeoptions.nlp.integrator.nodes = 2
    codeoptions.maxit = 30
    codeoptions.printlevel = 0
    codeoptions.optlevel = 2
    codeoptions.nlp.stack_parambounds = 2

    print("[SOLVERINFO] Generating DRL-enabled MPCC solver...")
    print(f"  Parameter dimension: {model.npar} (standard: 17, added: 6 for DRL)")
    print(f"  New parameters: a_ref, delta_ref, omega, K_a, K_delta, beta")

    solver = model.generate_solver(codeoptions)
    print("[SOLVERINFO] DRL-enabled solver generated successfully!")

    return solver
