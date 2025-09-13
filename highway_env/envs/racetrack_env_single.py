from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle


class RacetrackEnvSingle(AbstractEnv):

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
                "policy_frequency": 20,
                "duration": 100,
                # "collision_reward": -1,
                "lane_centering_cost": 4,
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
            }
        )
        return config

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

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight,
        # Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
            width=5,
            speed_limit=speedlimits[1],
        )
        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)

        # net.add_lane(
        #     "a",
        #     "b",
        #     StraightLane(
        #         [42, 5],
        #         [100, 5],
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         width=5,
        #         speed_limit=speedlimits[1],
        #     ),
        # )

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
        # net.add_lane(
        #     "b",
        #     "c",
        #     CircularLane(
        #         center1,
        #         radii1 + 5,
        #         np.deg2rad(90),
        #         np.deg2rad(-1),
        #         width=5,
        #         clockwise=False,
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         speed_limit=speedlimits[2],
        #     ),
        # )

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
        # net.add_lane(
        #     "c",
        #     "d",
        #     StraightLane(
        #         [125, -20],
        #         [125, -30],
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         width=5,
        #         speed_limit=speedlimits[3],
        #     ),
        # )

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
        # net.add_lane(
        #     "d",
        #     "e",
        #     CircularLane(
        #         center2,
        #         radii2 + 5,
        #         np.deg2rad(0),
        #         np.deg2rad(-181),
        #         width=5,
        #         clockwise=False,
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         speed_limit=speedlimits[4],
        #     ),
        # )

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
        # net.add_lane(
        #     "e",
        #     "f",
        #     CircularLane(
        #         center3,
        #         radii3,
        #         np.deg2rad(0),
        #         np.deg2rad(137),
        #         width=5,
        #         clockwise=True,
        #         line_types=(LineType.NONE, LineType.CONTINUOUS),
        #         speed_limit=speedlimits[5],
        #     ),
        # )

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
        # net.add_lane(
        #     "f",
        #     "g",
        #     StraightLane(
        #         [59.3934, -19.2],
        #         [39.3934, -39.2],
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         width=5,
        #         speed_limit=speedlimits[6],
        #     ),
        # )

        # 7 - Circular Arc #4 - Bugs out when arc is too large, thus 2 sections
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
        # net.add_lane(
        #     "g",
        #     "h",
        #     CircularLane(
        #         center4,
        #         radii4 + 5,
        #         np.deg2rad(315),
        #         np.deg2rad(165),
        #         width=5,
        #         clockwise=False,
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         speed_limit=speedlimits[7],
        #     ),
        # )
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
        # net.add_lane(
        #     "h",
        #     "i",
        #     CircularLane(
        #         center4,
        #         radii4 + 5,
        #         np.deg2rad(170),
        #         np.deg2rad(58),
        #         width=5,
        #         clockwise=False,
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         speed_limit=speedlimits[7],
        #     ),
        # )

        # 8 - Circular Arc #5 - Reconnects to Start
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
        # net.add_lane(
        #     "i",
        #     "a",
        #     CircularLane(
        #         center5,
        #         radii5,
        #         np.deg2rad(238),
        #         np.deg2rad(268),
        #         width=5,
        #         clockwise=True,
        #         line_types=(LineType.NONE, LineType.CONTINUOUS),
        #         speed_limit=speedlimits[8],
        #     ),
        # )

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
                speed=None,
                longitudinal=rng.uniform(30, 80),  # 从(20, 50)调整到(30, 80)
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
