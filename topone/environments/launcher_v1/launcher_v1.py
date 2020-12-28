from math import pi

from dataclasses import dataclass
from typing import Sequence, Optional, Tuple

import gym
import numpy as np
import numba as nb

from .simulation import Simulation


class Stage:
    def __init__(self,
                 dry_mass: float,
                 propellant_mass: float,
                 specific_impulse: float,
                 thrust: float,
                 n_ignitions: Optional[int] = 1):
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.specific_impulse = specific_impulse
        self.thrust = thrust
        self.n_ignitions = n_ignitions

    def to_tuple(self):
        return (np.float64(self.dry_mass),
                np.float64(self.propellant_mass),
                np.float64(self.specific_impulse),
                np.float64(self.thrust),
                np.int32(-1 if self.n_ignitions is None else self.n_ignitions))


class LauncherV1(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self,
                 surface_diameter: float,
                 mu: float,
                 stages: Sequence[Stage],
                 initial_longitude: float,
                 initial_altitude: float,
                 initial_theta_e: float):

        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(2),  # Engine command
            gym.spaces.Discrete(2),  # Drop stage command
            gym.spaces.Box(low=-pi, high=pi, shape=(1,)),  # Flight path angle command
        ))
        self.observation_space = gym.spaces.Dict()

        self.random = np.random.Generator(np.random.PCG64())

        self.sim = Simulation(
            surface_diameter=surface_diameter,
            mu=mu,
            stages=nb.typed.List([stage.to_tuple() for stage in stages]),
            initial_longitude=initial_longitude,
            initial_altitude=initial_altitude,
            initial_theta_e=initial_theta_e,
        )

        self.reset()

    def reset(self):
        self.sim.reset()

    def seed(self, seed=None):
        self.random = np.random.Generator(np.random.PCG64(seed))

    def step(self, action: Tuple[bool, bool, float]):
        self.sim.step((
            action[0],
            action[1],
            np.float64(action[2])
        ))

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def sim_states_dict(self):
        return {name: str(getattr(self.sim, name)) for name in state_names}


state_names = (
    "t", "command_engine_on", "command_drop_stage", "command_gamma_e", "pid_command_theta_e",
    "pid_command_gamma_e_dot", "gii", "xii", "vii", "aii", "tei", "vie", "fii_thrust", "theta_i", "theta_i_dot",
    "theta_e", "mass", "mass_dot", "h", "engine_on", "stage_state", "stage_idx", "stage_ignitions_left",
    "gamma_i",
    "gamma_e", "longitude", "reward", "score", "done",
)
