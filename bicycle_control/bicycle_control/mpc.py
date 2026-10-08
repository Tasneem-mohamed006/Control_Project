"""
High-Level Lateral Steering Controller: Extended Kinematic Bicycle MPC.
Solves a constrained non-linear program over prediction horizon N using SciPy,
optimizing steering angle and longitudinal acceleration (mapped to throttle).
"""

import math  # noqa: F401
import numpy as np  # noqa: F401
from scipy.optimize import minimize  # noqa: F401


class KinematicBicycleMPC:
    """Nonlinear Model Predictive Control for an Extended Kinematic Bicycle Model.

    Optimizes future control sequences u = [delta_k, a_k] where steering angle delta_k
    and longitudinal acceleration a_k (mapped to throttle effort) are the control inputs,
    forward-simulating a 4-state extended kinematic bicycle model x = [x, y, theta, v]^T.
    """

    def __init__(self, wheelbase=1.25, dt=0.1, horizon=10,
                 max_steer_rad=math.radians(35.0), k_a=4.0,
                 max_accel=None, max_brake=None):
        self.L = wheelbase
        self.dt = dt
        self.N = horizon
        self.max_steer_rad = max_steer_rad
        self.k_a = float(max_accel if max_accel is not None else k_a)

        # Weights: heavily penalize lateral CTE, heading error, and steering rate
        self.w_lat = 30.0
        self.w_long = 1.0
        self.w_yaw = 10.0
        self.w_v = 1.0
        self.w_steer = 0.2
        self.w_dsteer = 6.0
        self.w_accel = 0.1

        self.last_u = np.zeros(2 * self.N)  # warm-start [delta_0, a_0, delta_1, a_1, ...]

    def solve(self, x0, ref_trajectory, current_steer=0.0):
        """Solves MPC optimization problem over horizon N.

        x0: [x, y, yaw, v]
        ref_trajectory: list of length N containing [x_ref, y_ref, yaw_ref, v_ref]
        current_steer: actual current steering angle in radians
        Returns: (steer_rad, throttle_cmd in [-1.0, 1.0])
        """
        # ======================================================================
        # TODO: Milestone 5.4 — Extended Kinematic Bicycle MPC
        #
        # 1. Horizon & Bounds Setup:
        #    - Determine effective horizon N = min(self.N, len(ref_trajectory)).
        #    - If N < 2, return (0.0, 0.0).
        #    - Construct variable bounds for the decision vector:
        #      u = [delta_0, a_0, delta_1, a_1, ..., delta_N-1, a_N-1]
        #      where delta_k in [-self.max_steer_rad, self.max_steer_rad] (steering input)
        #      and a_k in [-self.k_a, self.k_a] (longitudinal acceleration input).
        #
        # 2. Objective Function objective(u):
        #    - Unpack state [x, y, yaw, v] from x0 and set prev_delta = current_steer.
        #    - For each horizon step k in 0 .. N-1:
        #        a. Forward simulate state using discrete Extended Kinematic Bicycle equations
        #           (where longitudinal velocity v is an explicit state variable integrated
        #           forward with acceleration input a_k)
        #        b. Project tracking error into the path-aligned Frenet frame.
        #        c. Accumulate weighted quadratic costs:
        #           lateral CTE, heading error, speed error, steering, slew rate, accel.
        #        d. Update prev_delta = delta_k.
        #    - Return total cost.
        #
        # 3. Warm-Start Initialization:
        #    - Construct u_init by shifting self.last_u forward by 1 time step.
        #
        # 4. Numerical Optimization & Control Extraction:
        #    - Call scipy.optimize.minimize(objective, u_init, bounds=bounds,
        #                                   method='SLSQP',
        #                                   options={'maxiter': 25, 'ftol': 1e-3}).
        #    - Save optimal solution in self.last_u.
        #    - Extract first control step: delta_cmd = u*[0], accel_cmd = u*[1].
        #    - Map optimal acceleration a_0* to normalized throttle in [-1.0, 1.0]:
        #      throttle_cmd = accel_cmd / self.k_a
        #    - Return tuple: (delta_cmd, throttle_cmd).
        # ======================================================================

        N = min(self.N, len(ref_trajectory))
        if N < 2:
            return (0.0, 0.0)

    # ---- bounds: u = [delta_0, a_0, delta_1, a_1, ...]
        bounds = [(-self.max_steer_rad, self.max_steer_rad),
                  (-self.k_a, self.k_a)] * N

    # ---- precompute everything that does not depend on u
        ref = np.asarray(ref_trajectory[:N], dtype=float)
        xr, yr, yawr, vr = (ref[:, i].tolist() for i in range(4))
        sin_r = np.sin(yawr).tolist()
        cos_r = np.cos(yawr).tolist()

        dt, L = self.dt, self.L
        cos, sin, tan = math.cos, math.sin, math.tan
        pi, two_pi = math.pi, 2.0 * math.pi
        x_init, y_init, yaw_init, v_init = map(float, x0)

        def objective(u):
            x, y, yaw, v = x_init, y_init, yaw_init, v_init
            prev_delta = current_steer
            cost = 0.0

            for k in range(N):
                delta = u[2 * k]
                accel = u[2 * k + 1]

            # extended kinematic bicycle, forward Euler
                x   += v * cos(yaw) * dt
                y   += v * sin(yaw) * dt
                yaw += v / L * tan(delta) * dt
                v    = max(0.0, v + accel * dt)

            # Frenet-frame errors
                dx, dy = x - xr[k], y - yr[k]
                e_lat = -sin_r[k] * dx + cos_r[k] * dy
                e_yaw = (yaw - yawr[k] + pi) % two_pi - pi
                e_v   = v - vr[k]
                d_delta = delta - prev_delta

                cost += (self.w_lat    * e_lat ** 2
                       + self.w_yaw    * e_yaw ** 2
                       + self.w_v      * e_v ** 2
                       + self.w_steer  * delta ** 2
                       + self.w_dsteer * d_delta ** 2
                       + self.w_accel  * accel ** 2)
                prev_delta = delta

            return cost

    # ---- warm start: shift previous solution by one step, repeat the last
        prev = np.asarray(self.last_u, dtype=float)
        if prev.size < 2 * N:                       # safety if N grew / first call
            prev = np.concatenate([prev, np.zeros(2 * N - prev.size)])
        prev = prev[:2 * N]
        u_init = np.concatenate([prev[2:], prev[-2:]])

        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        u_init = np.clip(u_init, lo, hi)

        res = minimize(objective, u_init, bounds=bounds, method='SLSQP',
                   options={'maxiter': 25, 'ftol': 1e-3})

        self.last_u = res.x
        return (float(res.x[0]), float(res.x[1] / self.k_a))