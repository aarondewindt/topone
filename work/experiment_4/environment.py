from math import radians
import random
from typing import Any, Tuple

from gym import spaces
import numpy as np

from cw.simulation import GymEnvironment, alias_states
from cw.control import PController, PIDController


hpi = np.pi / 2


class Environment(GymEnvironment):
    def __init__(self,
                 required_states=None,
                 target_time_step=None
                 ):
        required_states = required_states or []
        super().__init__(
            target_time_step=target_time_step,
            required_states=required_states + [
                "command_theta_e",
                "command_gamma_e"
            ],
        )

        self.gamma_controller = PIDController(4, 0.1, 0.1)
        self.kp_theta = 10

        # Environment spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(1),
            spaces.Discrete(1),
            spaces.Discrete(1),
            spaces.Box(low=np.inf, high=np.inf, shape=(1,))
        ))

    def initialize(self,  simulation):
        super().initialize(simulation)
        self.gamma_controller.reset()
        initial_theta_e = hpi + radians(random.uniform(-20, 20))
        simulation.states.theta = -hpi + simulation.states.latitude + initial_theta_e

    @alias_states
    def environment_step(self, is_last):
        self.s.command_gamma_e = hpi

        if self.s.t > 0.1:
            self.gamma_controller.command = self.s.command_gamma_e
            self.s.command_theta_e = self.s.theta_e + self.gamma_controller.step(self.s.t, self.s.gamma_e)
            self.s.theta_dot = np.clip(self.kp_theta * (self.s.command_theta_e - self.s.theta_e), -1, 1)
        else:
            self.s.command_gamma_e = self.s.gamma_e
            self.s.command_theta_e = self.s.theta_e
            self.s.theta_dot = 0.

        self.s.command_engine_on = True
        if self.s.stage_state == 2:
            self.s.command_drop_stage = True

    @alias_states
    def act(self, action: Any):
        pass

    @alias_states
    def observe(self, done: bool) -> Tuple[Any, float, dict]:
        observation = np.array((
            self.s.stage_state == 0,
            self.s.stage_state == 1,
            self.s.stage_state == 2,
            self.s.h
        ))

        reward = self.s.h if done else self.s.h
        info = {'time': self.s.t}

        if self.s.t > 1 and (self.s.vic[1] <= 0):
            self.simulation.stop()

        return observation, reward, info
