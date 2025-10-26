from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


class RacetrackEnvSinglePursuit(AbstractEnv):
    """
    Racetrack environment using Pure Pursuit controller.

    The agent only controls the target speed (FASTER/SLOWER),
    while steering is automatically handled by Pure Pursuit algorithm.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "features": ["x", "y", "vx", "vy", "heading"],
                    "absolute": False,
                    "normalize": True,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [7, 9, 10, 12],  # Discrete speed options (increased minimum)
                    "longitudinal": True,
                    "lateral": True,  # No lane changes (single lane)
                },
                "simulation_frequency": 20,
                "policy_frequency": 10,
                "duration": 100,
                "lane_centering_cost": 4,
                "progress_reward": 1.0,
                "speed_reward": 1.0,
                "controlled_vehicles": 1,
                "other_vehicles": 0,
                "screen_width": 800,
                "screen_height": 800,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
                "terminate_off_road": True,
            }
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(reward for reward in rewards.values())
        reward = utils.lmap(reward, [0, 5], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:
        # Get lateral offset from lane center
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)

        # Determine optimal speed based on lane type
        from highway_env.road.lane import CircularLane, StraightLane
        lane = self.vehicle.lane

        if isinstance(lane, CircularLane):
            # In curves: reward slower speeds based on curve radius
            if lane.radius < 20:  # Tight curves (like curve 4 with radius 15)
                optimal_speed = 8.0
            else:
                optimal_speed = 9.0
        else:  # StraightLane
            # In straights: reward higher speeds
            optimal_speed = 12.0

        # Longitudinal speed along the lane
        longitudinal_speed = self.vehicle.speed * np.cos(
            self.vehicle.heading
            - self.vehicle.lane.heading_at(
                self.vehicle.lane.local_coordinates(self.vehicle.position)[0]
            )
        )

        # Speed error relative to optimal speed
        if longitudinal_speed >= 0:
            speed_error = abs(optimal_speed - longitudinal_speed)
        else:
            speed_error = abs(longitudinal_speed) + optimal_speed

        progress_reward = 1 / (1 + self.config["progress_reward"] * speed_error**2)

        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "progress_reward": progress_reward,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        if self.config["terminate_off_road"]:
            return not self.vehicle.on_road
        return False

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Same track as racetrack_env_single
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # 1 - Straight Section
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
            width=5,
            speed_limit=speedlimits[1],
        )
        net.add_lane("a", "b", lane)

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 3 - Vertical Straight
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -30],
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                speed_limit=speedlimits[5],
            ),
        )

        # 6 - Slant
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [55.7, -15.7],
                [35.7, -35.7],
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )

        # 7 - Circular Arc #4 (split into 2 sections)
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(315),
                np.deg2rad(170),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(170),
                np.deg2rad(56),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )

        # 8 - Circular Arc #5 (reconnects to start)
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(240),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Create ego vehicle with Pure Pursuit controller.
        """
        rng = self.np_random

        # Controlled vehicle (ControlledVehicle for Pure Pursuit)
        self.controlled_vehicles = []
        lane_index = ("a", "b", 0)

        # Use ControlledVehicle instead of regular Vehicle
        controlled_vehicle = self.action_type.vehicle_class(
            self.road,
            self.road.network.get_lane(lane_index).position(rng.uniform(5, 30), 0),
            speed=5.0,
            heading=self.road.network.get_lane(lane_index).heading_at(10),
        )

        # Set initial target speed
        controlled_vehicle.target_speed = 9.0

        # Plan the complete route around the track
        controlled_vehicle.route = [
            ("a", "b", 0),
            ("b", "c", 0),
            ("c", "d", 0),
            ("d", "e", 0),
            ("e", "f", 0),
            ("f", "g", 0),
            ("g", "h", 0),
            ("h", "i", 0),
            ("i", "a", 0),
        ]

        # Set initial target lane
        controlled_vehicle.target_lane_index = lane_index

        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)
