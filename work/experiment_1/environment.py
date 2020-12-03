from gym import spaces

from topone.environment_base import EnvironmentBase


class Environment(EnvironmentBase):
    @property
    def action_space(self) -> spaces.Space:
        return spaces.Discrete(2)

    def step(self):
        self.s.reward = (self.s.h/1000)**3
        if (self.s.stage_state == 2) and (self.s.vic[1] < 0):
            self.s.done = True
            self.simulation.stop()
        else:
            self.s.done = False

    def act(self, action):
        self.simulation.states.command_engine_on = bool(action)
