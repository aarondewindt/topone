from math import atan2
from enum import IntEnum
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


class Action(IntEnum):
    engine_on = 0
    engine_off = 1
    drop_stage = 2


class IdealAgent(ModuleBase):
    def __init__(self, *,
                 target_time_step=0.1,):
        super().__init__(
            is_discreet=True,
            target_time_step=target_time_step,
            required_states=[
                "stage_state",
                "stage_idx",
                "reward",

                "command_engine_on",
                "command_drop_stage",
                "delta_v"
            ],
        )

    def initialize(self, simulation):
        super().initialize(simulation)

    def store(self):
        pass

    def load_last(self, missing_ok=False):
        pass

    def load_idx(self, idx):
        pass

    def load(self, path):
        pass

    def clean(self):
        pass

    def greedy_action(self, s) -> Action:
        if s.stage_state == UNFIRED:
            return Action.engine_on
        elif s.stage_state == FIRING:
            return Action.engine_on
        else:
            return Action.drop_stage

    def evaluate_policy(self, s) -> Action:
        return self.greedy_action(s)

    def display_value(self):
        pass

    def display_greedy_policy(self):
        class States:
            def __init__(self, stage_state, stage_idx):
                self.stage_state = stage_state
                self.stage_idx = stage_idx

        contents = f"Stage 0\n" \
                   f"  UNFIRED: {self.greedy_action(States(UNFIRED, 0)).name} \n" \
                   f"  FIRING: {self.greedy_action(States(FIRING, 0)).name}\n" \
                   f"  FIRED: {self.greedy_action(States(FIRED, 0)).name}\n" \
                   f"Stage 1\n" \
                   f"  UNFIRED: {self.greedy_action(States(UNFIRED, 1)).name} \n" \
                   f"  FIRING: {self.greedy_action(States(FIRING, 1)).name}\n" \
                   f"  FIRED: {self.greedy_action(States(FIRED, 1)).name}"
        print(contents)

    def step(self):
        # Unpack values from the previous step (t0).
        s = self.s

        action_t1 = self.evaluate_policy(s)

        # Perform action
        if action_t1 == Action.engine_on:
            s.command_engine_on = True
            s.command_drop_stage = False
        elif action_t1 == Action.engine_off:
            s.command_engine_on = False
            s.command_drop_stage = False
        else:  # Drop stage
            s.command_engine_on = False
            s.command_drop_stage = True




