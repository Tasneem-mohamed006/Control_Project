"""
High-Level Lateral Steering Controller: Geometric Pure Pursuit.
Calculates steering curvature from lookahead arc geometry.
"""

import math  # noqa: F401
import numpy as np  # noqa: F401


class PurePursuitController:
    """Adaptive Pure Pursuit lateral controller."""

    def __init__(self, wheelbase=1.25, kv=0.25, l_min=0.8, l_max=2.5,
                 max_steer_rad=math.radians(35.0)):
        self.L = wheelbase
        self.kv = kv
        self.l_min = l_min
        self.l_max = l_max
        self.max_steer_rad = max_steer_rad

    def compute_lookahead(self, v):
        """Adaptive lookahead distance: Ld = clip(kv * v + l_min, l_min, l_max)."""
        # TODO: Milestone 5.3 Step 1 — Adaptive Lookahead Horizon
        # The car looks further ahead at higher speeds to plan smoother turns.
        # Implement the speed-scaled lookahead formula and clamp it to the allowed range.
        lookahead = self.kv * v + self.l_min

        lookahead = np.clip(
            lookahead,
            self.l_min,
            self.l_max
        )

        return float(lookahead)

    def find_target_waypoint(self, x, y, path_points, lookahead):
        """Searches along path for the target waypoint at lookahead distance."""
        # TODO: Milestone 5.3 Step 2 — Target Waypoint Selection
        # This selects the goal point the car will steer toward.
        # Find the nearest waypoint on the path, then walk forward until
        # you reach one that is at least 'lookahead' meters away.
        # Find the nearest waypoint on the path
        distances = [
            math.hypot(px - x, py - y)
            for px, py, _ in path_points
        ]

        nearest_idx = int(np.argmin(distances))

        # Search forward from the nearest waypoint
        for i in range(nearest_idx, len(path_points)):
            px, py, _ = path_points[i]

            distance = math.hypot(px - x, py - y)

            if distance >= lookahead:
                return i, (px, py)

        last_idx = len(path_points) - 1
        px, py, _ = path_points[last_idx]

        return last_idx, (px, py)

    def compute_steering(self, x, y, yaw, target_pt, lookahead):
        """Computes steering angle in radians using Pure Pursuit geometry."""
        # TODO: Milestone 5.3 Steps 3 & 4 — Coordinate Transformation & Arc Law
        # This is the core of Pure Pursuit: transform the target into the vehicle's
        # local frame, then use the arc geometry formula to compute the steering angle.
        target_x, target_y = target_pt

        dx = target_x - x
        dy = target_y - y

        # Rotate from world frame to vehicle frame
        x_local = math.cos(yaw) * dx + math.sin(yaw) * dy
        y_local = -math.sin(yaw) * dx + math.cos(yaw) * dy
        # Pure Pursuit curvature
        curvature = (2.0 * y_local) / (lookahead ** 2)

        # Convert curvature to steering angle
        steering = math.atan(self.L * curvature)

        # Clamp steering
        steering = np.clip(
            steering,
            -self.max_steer_rad,
            self.max_steer_rad
        )

        return float(steering)
