import numpy as np

from gym import spaces

from topone.environment_base import EnvironmentBase


class Environment(EnvironmentBase):
    @property
    def action_space(self) -> spaces.Space:
        return spaces.Discrete(3)

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Tuple((spaces.Discrete(2),
                             spaces.Discrete(2),
                             spaces.Discrete(2),
                             spaces.Discrete(2),
                             spaces.Discrete(2),
                             spaces.Box(low=np.inf, high=np.inf, shape=(1,))
                             ))

    def observe(self):
        s = self.simulation.states
        return (int(s.stage_state == 0),
                int(s.stage_state == 1),
                int(s.stage_state == 2),
                int(s.stage_idx == 0),
                int(s.stage_idx == 1),
                s.h)

    def act(self, action):
        s = self.simulation.states
        if action == 0:
            s.command_engine_on = False
            s.command_drop_stage = False
        elif action == 1:
            s.command_engine_on = True
            s.command_drop_stage = False
        else:
            s.command_engine_on = False
            s.command_drop_stage = True

    def step(self):
        # self.s.reward = (self.s.h/1000)**3

        # self.s.reward = self.s.vic[1]

        self.s.reward = 0

        if self.s.t > 1 and (self.s.vic[1] <= 0):
            self.simulation.stop()

    def end(self):
        self.simulation.states.reward = self.simulation.states.vic[1]
