from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle


class RacetrackEnvComplex(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-30, 30], [-30, 30]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
                "simulation_frequency": 20,
                "policy_frequency": 10,  # Reduced from 20 to 10 Hz to reduce action oscillation
                "duration": 100,
                # "collision_reward": -1,
                "lane_centering_cost": 10,
                "progress_reward": 0.5,
                "speed_reward": 1,
                "action_smoothness_cost": 1,
                "progress_cost": 1,
                "controlled_vehicles": 1,
                "other_vehicles": 0,
                "screen_width": 800,
                "screen_height": 800,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
                "terminate_off_road": True,
                # Action smoothing parameters
                "action_smoothing_enabled": True,
                "action_smoothing_alpha": 0.0,  # Pure rate limiting (no EMA memory)
                "action_smoothing_rate_limit": 0.2,  # Max 30% change per step
            }
        )
        return config

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing and rate limiting to action.

        ã_t = α·ã_{t-1} + (1-α)·a_t
        subject to |ã_t - ã_{t-1}| ≤ Δ_max
        """
        if not self.config.get("action_smoothing_enabled", False):
            return action

        action = np.array(action)

        # First step: no smoothing
        if self.previous_action_smooth is None:
            self.previous_action_smooth = action.copy()
            return action

        alpha = self.config["action_smoothing_alpha"]
        rate_limit = self.config["action_smoothing_rate_limit"]

        # Step 1: EMA smoothing
        action_ema = alpha * self.previous_action_smooth + (1 - alpha) * action

        # Step 2: Rate limiting
        # Assuming action space is [-1, 1], range is 2
        action_range = 2.0
        delta_max = rate_limit * action_range

        delta = action_ema - self.previous_action_smooth
        delta_clipped = np.clip(delta, -delta_max, delta_max)

        action_smooth = self.previous_action_smooth + delta_clipped

        # Ensure within bounds [-1, 1]
        action_smooth = np.clip(action_smooth, -1.0, 1.0)

        # Update
        self.previous_action_smooth = action_smooth.copy()

        return action_smooth

    def step(self, action):
        """Override step to add action smoothing."""
        # Store raw action for potential use
        self.raw_action = np.array(action)

        # Apply smoothing
        action_smooth = self._smooth_action(action)

        # Execute with smoothed action
        # NOTE: The reward will be computed based on action_smooth (passed to _reward)
        return super().step(action_smooth)

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(reward for reward in rewards.values())
        reward = utils.lmap(reward, [0, 5], [0, 1])
        reward *= rewards["on_road_reward"]
        reward *= rewards["collision_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:

        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)

        target_speed = self.vehicle.lane.speed_limit * 0.95
        # speed_diff = abs(self.vehicle.speed - target_speed)

        longitudinal_speed = self.vehicle.speed * np.cos(
            self.vehicle.heading
            - self.vehicle.lane.heading_at(
                self.vehicle.lane.local_coordinates(self.vehicle.position)[0]
            )
        )

        if longitudinal_speed >= 0:
            # Fixed: Use abs() to penalize both under and overspeeding
            progress_diff = max(0, target_speed - longitudinal_speed)
        else:
            progress_diff = abs(longitudinal_speed) + target_speed

        progress_reward = 1 / (1 + self.config["progress_cost"] * progress_diff**2)

        action_smoothness_reward = 1.0  # first step
        if self.previous_action is not None:
            action_diff = np.linalg.norm(action - self.previous_action)
            action_smoothness_reward = 1 / (
                1 + self.config["action_smoothness_cost"] * action_diff**2
            )

        # update
        self.previous_action = action.copy()

        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            # "speed_reward": 1 / (1 + self.config["speed_reward"] * speed_diff**2),
            "progress_reward": progress_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "collision_reward": not self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        if self.config["terminate_off_road"]:
            return self.vehicle.crashed or not self.vehicle.on_road
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.previous_action = None
        # Reset action smoothing
        self.previous_action_smooth = None

    @staticmethod
    def loadTrack_03():
        """Load track definition in simplified format."""
        x0 = np.array([0, 0]).reshape(-1, 1)  # Starting position
        th0 = 0  # Starting angle
        w = 12  # Track width
        trackdata = [
            ['s', 14 * 3],      # Straight: 42m
            ['c', [15 * 3, -90]],   # Curve: radius 45, right turn 90°
            ['s', 5 * 3],       # Straight: 15m
            ['c', [4 * 3, 90]],     # Curve: radius 12, left turn 90°
            ['c', [4 * 3, -90]],    # Curve: radius 12, right turn 90°
            ['s', 5 * 3],       # Straight: 15m
            ['c', [3.5 * 3, -90]],  # Curve: radius 10.5, right turn 90°
            ['s', 16 * 3],      # Straight: 48m
            ['c', [3.5 * 3, -120]], # Curve: radius 10.5, right turn 120°
            ['s', 10 * 3],      # Straight: 30m
            ['c', [10 * 3, 120]],   # Curve: radius 30, left turn 120°
            ['s', 10 * 3],      # Straight: 30m
            ['c', [5 * 3, 90]],     # Curve: radius 15, left turn 90°
            ['s', 5 * 3],       # Straight: 15m
            ['c', [5 * 3, 90]],     # Curve: radius 15, left turn 90°
            ['c', [7 * 3, -180]],   # Curve: radius 21, right turn 180°
            ['s', 2.3 * 3],     # Straight: 6.9m
            ['c', [10 * 3, -90]],   # Curve: radius 30, right turn 90°
            ['s', 14.6 * 3],    # Straight: 43.8m
            ['c', [12 * 3, -90]]    # Curve: radius 36, right turn 90°
        ]
        return trackdata, x0, th0, w

    def _make_road(self) -> None:
        """Create road network from trackdata format."""
        net = RoadNetwork()

        # Load track definition
        trackdata, x0, th0, w = self.loadTrack_03()

        # Current position and heading
        r_IV = x0  # Current position as column vector
        th0_current = th0  # Current heading angle

        def A_z(th):
            """Rotation matrix for angle th."""
            return np.array([[np.cos(th), -np.sin(th)],
                            [np.sin(th), np.cos(th)]])

        A_IV = A_z(th0)  # Current direction matrix

        # Track adaptive speed limits based on segment type
        segment_speed_limits = []
        for segment in trackdata:
            if segment[0] == 's':  # Straight
                segment_speed_limits.append(12.0)
            elif segment[0] == 'c':  # Curve
                radius = segment[1][0]
                # Lower speed for tighter curves
                if radius < 15:
                    segment_speed_limits.append(8.0)
                elif radius < 25:
                    segment_speed_limits.append(9.0)
                else:
                    segment_speed_limits.append(10.0)

        # Generate lanes from trackdata
        for idx, segment in enumerate(trackdata):
            # Generate lane names
            start_node = chr(ord('a') + idx)
            end_node = chr(ord('a') + (idx + 1) % len(trackdata))

            speed_limit = segment_speed_limits[idx]

            if segment[0] == 's':  # Straight segment
                dist = segment[1]

                # Start and end positions
                start_pos = r_IV.flatten()
                end_pos = (r_IV + dist * A_IV[:, 0].reshape(-1, 1)).flatten()

                lane = StraightLane(
                    start_pos.tolist(),
                    end_pos.tolist(),
                    line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                    width=w,
                    speed_limit=speed_limit,
                )
                net.add_lane(start_node, end_node, lane)

                # Update position (following original algorithm)
                r_IV = r_IV + dist * A_IV[:, 0].reshape(-1, 1)

            elif segment[0] == 'c':  # Circular segment
                rad = segment[1][0]  # radius
                ang = np.deg2rad(segment[1][1])  # angle in radians

                # Generate arc in local frame, then transform to global


                if ang > 0:  # Left turn (counter-clockwise)
                    # In local frame: generate arc from 0 to ang
                    # After rotation, center is at (0, rad) in local frame
                    arc_end_local = rad * np.array([[np.cos(ang)], [np.sin(ang)]])
                    # Apply the transformation matrix from original code
                    A_local = np.array([[0, 1], [-1, 0]])
                    center_offset_local = np.array([[0], [rad]])
                    track_end_local = A_local @ arc_end_local + center_offset_local

                    # Center in global frame
                    center_local = np.array([[0], [rad]])
                    center_global = A_IV @ center_local + r_IV

                    # Angles in global frame
                    start_angle = th0_current - np.pi/2
                    end_angle = start_angle + ang
                    clockwise = False

                else:  # Right turn (clockwise)
                    # In local frame: generate arc from 0 to ang (negative)
                    # After rotation, center is at (0, -rad) in local frame
                    arc_end_local = rad * np.array([[np.cos(ang)], [np.sin(ang)]])
                    # Apply the transformation matrix from original code
                    A_local = np.array([[0, -1], [1, 0]])
                    center_offset_local = np.array([[0], [-rad]])
                    track_end_local = A_local @ arc_end_local + center_offset_local

                    # Center in global frame
                    center_local = np.array([[0], [-rad]])
                    center_global = A_IV @ center_local + r_IV

                    # Angles in global frame
                    start_angle = th0_current + np.pi/2
                    end_angle = start_angle + ang
                    clockwise = True

                lane = CircularLane(
                    center_global.flatten().tolist(),
                    rad,
                    start_angle,
                    end_angle,
                    width=w,
                    clockwise=clockwise,
                    line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                    speed_limit=speed_limit,
                )
                net.add_lane(start_node, end_node, lane)

                # Update position FIRST using old A_IV 
                r_IV = A_IV @ track_end_local + r_IV

                # Then update heading and direction matrix
                th0_current = th0_current + ang
                A_IV = A_z(ang) @ A_IV

        # Create road object
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and
        on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", 0) if i == 0 else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index,
                speed=5.0,
                longitudinal=rng.uniform(5, 30),
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # Front vehicle
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            self.road.vehicles.append(vehicle)

            # Other vehicles
            for i in range(rng.integers(self.config["other_vehicles"])):
                rand_lane_index = self.road.network.random_lane_index(rng)

                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    rand_lane_index,
                    longitudinal=rng.uniform(
                        low=0, high=self.road.network.get_lane(rand_lane_index).length
                    ),
                    speed=6 + rng.uniform(high=3),
                )
                # Prevent early collisions
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)
