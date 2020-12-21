from random import random
from typing import Optional
from pathlib import Path

import numpy as np

from topone.dynamics_1 import UNFIRED, FIRED, FIRING
from topone.agent_module_base import AgentModuleBase
from topone.environment_base import EnvironmentBase


class Agent(AgentModuleBase):
    def __init__(self, *,
                 epsilon: float,
                 alpha: float,
                 gamma: float,
                 path: Optional[Path]=None,
                 target_time_step=0.1,
                 load_last=True):
        super().__init__(
            path=path,
            target_time_step=target_time_step,
            required_states=[
                "stage_state",
                "reward",

                "command_engine_on",
                "delta_v"
            ],
        )

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.values = np.zeros((3, 2))
        self.state_action_t0 = None, None, None

        self.environment = None
        self.action_space = None

        if load_last:
            self.load_last(missing_ok=True)

    def get_metadata(self):
        return self.epsilon, self.alpha, self.gamma, self.values

    def set_metadata(self, metadata):
        self.epsilon, self.alpha, self.gamma, self.values = metadata

    def initialize(self, simulation):
        super().initialize(simulation)
        self.state_action_t0 = None, None, None
        self.environment = simulation.find_modules_by_type(EnvironmentBase)[0]
        self.action_space = self.environment.action_space

    def greedy_action(self, s):
        return int(self.values[s.stage_state, 1] > self.values[s.stage_state, 0])

    def evaluate_policy(self, s):
        if random() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.greedy_action(s)

    def display_greedy_policy(self):
        class States:
            def __init__(self, stage_state):
                self.stage_state = stage_state

        contents = f"UNFIRED: {self.greedy_action(States(UNFIRED))} \n" \
                   f"FIRING: {self.greedy_action(States(FIRING))}\n" \
                   f"FIRED: {self.greedy_action(States(FIRED))}"

        print(contents)

    def step(self):
        # Unpack values from the previous step (t0).
        s = self.s
        state_t0, action_t0, value_t0 = self.state_action_t0

        # Evaluate the policy for the current step (t1).
        state_t1 = s.stage_state
        action_t1 = self.evaluate_policy(s)
        value_t1 = self.values[state_t1, action_t1]

        # Store the values for the next step.
        self.state_action_t0 = state_t1, action_t1, value_t1

        if state_t0 is None:
            # Do not learn on the first step since we haven't taken any actions yet.
            s.delta_v = 0
        else:
            # Learn using Sarsa.
            s.delta_v = self.alpha * (s.reward + self.gamma * value_t1 - value_t0)
            self.values[state_t0, action_t0] += s.delta_v

        # Perform action
        s.command_engine_on = action_t1



