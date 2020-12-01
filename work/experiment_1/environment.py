from gym import spaces

from topone.environment_base import EnvironmentBase


class Environment(EnvironmentBase):
    @property
    def action_space(self) -> spaces.Space:
        return spaces.Discrete(2)

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Discrete(3)

    @property
    def observation(self):
        return self.simulation.states.stage_state

    def step(self):
        self.s.reward = (self.s.h-1000)**3
        self.s.done = False

    def act(self, action):
        self.s.command_engine_on = bool(action)
