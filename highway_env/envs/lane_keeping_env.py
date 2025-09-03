from __future__ import annotations

import copy

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import BicycleVehicle


class LaneKeepingEnv(AbstractEnv):
    """A lane keeping control task."""

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode)
        self.lane = None
        self.lanes = []
        self.trajectory = []
        self.interval_trajectory = []
        self.lpv = None
        self.episode_length = 0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "AttributesObservation",
                    "attributes": ["state", "derivative", "reference_state"],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
                "simulation_frequency": 15,
                "policy_frequency": 15,
                "state_noise": 0.05,
                "derivative_noise": 0.05,
                "screen_width": 600,
                "screen_height": 250,
                "scaling": 7,
                "centering_position": [0.4, 0.5],
                "duration": 120,  # Episode duration in seconds
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "progress_reward": 0.3,
                "speed_reward": 0.2,
                "speed_limit": 8.3,
            }
        )
        return config

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.lanes and not self.lane.on_lane(self.vehicle.position):
            self.lane = self.lanes.pop(0)
        self.store_data()
        if self.lpv:
            self.lpv.set_control(
                control=action.squeeze(-1), state=self.vehicle.state[[1, 2, 4, 5]]
            )
            self.lpv.step(1 / self.config["simulation_frequency"])

        self.action_type.act(action)
        obs = self.observation_type.observe()
        self._simulate()

        # Increment episode length
        self.episode_length += 1

        info = {}
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        return obs, reward, terminated, truncated, info

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:
        _, lateral = self.lane.local_coordinates(self.vehicle.position)

        # Progress reward based on forward speed along the lane
        longitudinal_speed = self.vehicle.speed * np.cos(
            self.vehicle.heading
            - self.lane.heading_at(
                self.lane.local_coordinates(self.vehicle.position)[0]
            )
        )
        progress_reward = max(0, longitudinal_speed / self.config["speed_limit"])

        # Speed reward - encourage maintaining target speed
        target_speed = self.config["speed_limit"] * 0.8
        speed_diff = abs(self.vehicle.speed - target_speed)
        speed_reward = max(0, 1 - speed_diff / target_speed)

        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "progress_reward": progress_reward,
            "speed_reward": speed_reward,
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return False

    def _is_truncated(self) -> bool:
        # Truncate after specified duration (duration * frequency = number of steps)
        max_steps = self.config["duration"] * self.config["simulation_frequency"]
        return self.episode_length >= max_steps

    def _reset(self) -> None:
        self.episode_length = 0
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()
        lane = SineLane(
            [0, 0],
            [500, 0],
            amplitude=5,
            pulsation=2 * np.pi / 100,
            phase=0,
            width=10,
            line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS],
        )
        net.add_lane("a", "b", lane)
        other_lane = StraightLane(
            [50, 50],
            [115, 15],
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
            width=10,
        )
        net.add_lane("c", "d", other_lane)
        self.lanes = [other_lane, lane]
        self.lane = self.lanes.pop(0)
        net.add_lane(
            "d",
            "a",
            StraightLane(
                [115, 15],
                [115 + 20, 15 + 20 * (15 - 50) / (115 - 50)],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=10,
            ),
        )
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("c", "d", 0)).position(50, 0),
            heading=road.network.get_lane(("c", "d", 0)).heading_at(0),
            speed=8.3,
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    @property
    def dynamics(self) -> BicycleVehicle:
        return self.vehicle

    @property
    def state(self) -> np.ndarray:
        if not self.vehicle:
            return np.zeros((4, 1))
        return self.vehicle.state[[1, 2, 4, 5]] + self.np_random.uniform(
            low=-self.config["state_noise"],
            high=self.config["state_noise"],
            size=self.vehicle.state[[0, 2, 4, 5]].shape,
        )

    @property
    def derivative(self) -> np.ndarray:
        if not self.vehicle:
            return np.zeros((4, 1))
        return self.vehicle.derivative[[1, 2, 4, 5]] + self.np_random.uniform(
            low=-self.config["derivative_noise"],
            high=self.config["derivative_noise"],
            size=self.vehicle.derivative[[0, 2, 4, 5]].shape,
        )

    @property
    def reference_state(self) -> np.ndarray:
        if not self.vehicle or not self.lane:
            return np.zeros((4, 1))
        longi, lat = self.lane.local_coordinates(self.vehicle.position)
        psi_l = self.lane.heading_at(longi)
        state = self.vehicle.state[[1, 2, 4, 5]]
        return np.array([[state[0, 0] - lat], [psi_l], [0], [0]])

    def store_data(self) -> None:
        if self.lpv:
            state = self.vehicle.state.copy()
            interval = []
            for x_t in self.lpv.change_coordinates(
                self.lpv.x_i_t, back=True, interval=True
            ):
                # lateral state to full state
                np.put(state, [1, 2, 4, 5], x_t)
                # full state to absolute coordinates
                interval.append(state.squeeze(-1).copy())
            self.interval_trajectory.append(interval)
        self.trajectory.append(copy.deepcopy(self.vehicle.state))
