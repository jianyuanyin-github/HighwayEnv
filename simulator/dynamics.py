import casadi
import numpy as np
import yaml


class dynamics_simulator:
    def __init__(self, vehicleparams, Ts, x0, nodes):

        self.xvars = ["posx", "posy", "phi", "vx", "vy", "omega", "a", "delta", "theta"]
        self.uvars = ["jerk", "deltadot", "thetadot"]

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

        x = casadi.SX.sym("x", len(self.xvars))
        u = casadi.SX.sym("u", len(self.uvars))

        # extract states and inputs
        posx = x[self.xvars.index("posx")]
        posy = x[self.xvars.index("posy")]
        phi = x[self.xvars.index("phi")]
        vx = x[self.xvars.index("vx")]
        vy = x[self.xvars.index("vy")]
        omega = x[self.xvars.index("omega")]
        a = x[self.xvars.index("a")]
        delta = x[self.xvars.index("delta")]
        theta = x[self.xvars.index("theta")]

        jerk = u[self.uvars.index("jerk")]
        deltadot = u[self.uvars.index("deltadot")]
        thetadot = u[self.uvars.index("thetadot")]

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
            jerk,
            deltadot,
            thetadot,
        )

        # xdot = f(x,u)
        self.f = casadi.Function("f", [x, u], [statedot])

        # state of system
        self.x = x0
        # sampling timestep
        self.Ts = Ts
        # integration nodes
        self.nodes = nodes

    def tick(self, u):
        T_int = self.Ts / self.nodes
        for idx in range(self.nodes):
            self.x = self._integrate(T_int, u)
        self.wrap_phi()
        return self.x

    def set_theta(self, theta):
        self.x[self.xvars.index("theta")] = theta

    def wrap_phi(self):
        phi_idx = self.xvars.index("phi")
        while self.x[phi_idx] > np.pi:
            self.x[phi_idx] -= 2 * np.pi
        while self.x[phi_idx] < -np.pi:
            self.x[phi_idx] += 2 * np.pi

    # RK4 integration
    def _integrate(self, Ts, u):

        k1 = (
            self.f(self.x, u)
            .__array__()
            .reshape(
                -1,
            )
        )
        k2 = (
            self.f(self.x + Ts / 2 * k1, u)
            .__array__()
            .reshape(
                -1,
            )
        )
        k3 = (
            self.f(self.x + Ts / 2 * k2, u)
            .__array__()
            .reshape(
                -1,
            )
        )
        k4 = (
            self.f(self.x + Ts * k3, u)
            .__array__()
            .reshape(
                -1,
            )
        )
        self.x = self.x + 1 / 6 * Ts * (k1 + 2 * k2 + 2 * k3 + k4).reshape(
            -1,
        )
        return self.x
