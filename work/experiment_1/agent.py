from math import atan2
from enum import Enum
from random import random, choice
from copy import deepcopy
from typing import Optional
from pathlib import Path
import pickle
import time

import numpy as np
from cw.simulation import ModuleBase
from cw.control import PIDController

from topone.environment import UNFIRED, FIRED, FIRING

ACTION_ENGINE_ON = True
ACTION_ENGINE_OFF = False


class Agent(ModuleBase):
    def __init__(self, *,
                 epsilon: float,
                 alpha: float,
                 gamma: float,
                 path: Optional[Path]=None,
                 target_time_step=0.1,
                 load_last=True):
        super().__init__(
            is_discreet=True,
            target_time_step=target_time_step,
            required_states=[
                "stage_state",
                "reward",

                "command_engine_on",
                "delta_v"
            ],
        )
        self.path = Path(path)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.values = np.zeros((3, 2))
        self.state_action_t0 = None, None, None
        self.last_backup_time = None

        if load_last:
            self.load_last(missing_ok=True)

    def initialize(self, simulation):
        super().initialize(simulation)
        self.state_action_t0 = None, None, None

    def store(self):
        # Create agent data restore directory if it doesn't exist.
        self.path.mkdir(exist_ok=True)

        # Create path of the first restore point.
        idx = 0
        agent_path = self.path / f"agent.{idx}.pickle"

        # Loop until we find one that doesn't exist
        while agent_path.exists():
            idx += 1
            agent_path = self.path / f"agent.{idx}.pickle"

        with agent_path.open("wb") as f:
            self.last_backup_time = time.time()
            pickle.dump((self.epsilon, self.alpha, self.gamma, self.values, self.last_backup_time), f)

    def load_last(self, missing_ok=False):
        # Create agent data restore directory if it doesn't exist.
        self.path.mkdir(exist_ok=True)

        # Get a list of all indices
        idxs = []
        for path in self.path.glob(f"agent.*.pickle"):
            idxs.append(int(path.suffixes[0][1:]))

        if idxs:
            # Open the agent file with the highest index and load data.
            agent_path = self.path / f"agent.{max(idxs)}.pickle"
            with agent_path.open("rb") as f:
                self.epsilon, self.alpha, self.gamma, self.values, self.last_backup_time = pickle.load(f)
        else:
            if not missing_ok:
                raise FileNotFoundError("No stored agent data found.")

    def load(self, idx):
        self.path.mkdir(exist_ok=True)
        agent_path = self.path / f"agent.{idx}.pickle"
        with agent_path.open("rb") as f:
            self.epsilon, self.alpha, self.gamma, self.values, self.last_backup_time = pickle.load(f)

    def clean(self):
        self.path.mkdir(exist_ok=True)
        for path in self.path.glob(f"agent.*.pickle"):
            path.unlink()

    def greedy_action(self, s):
        return \
            ACTION_ENGINE_ON \
            if self.values[s.stage_state, 1] > self.values[s.stage_state, 0] \
            else ACTION_ENGINE_OFF

    def evaluate_policy(self, s):
        if random() < self.epsilon:
            return choice([ACTION_ENGINE_ON, ACTION_ENGINE_OFF])
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
        value_t1 = self.values[state_t1, int(action_t1)]

        # Store the values for the next step.
        self.state_action_t0 = state_t1, action_t1, value_t1

        if state_t0 is None:
            # Do not learn on the first step since we haven't taken any actions yet.
            s.delta_v = 0
        else:
            # Learn using Sarsa.
            s.delta_v = self.alpha * (s.reward + self.gamma * value_t1 - value_t0)
            self.values[state_t0, int(action_t0)] += s.delta_v

        # Perform action
        s.command_engine_on = action_t1



