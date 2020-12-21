from typing import Any, Tuple

from gym import spaces
import numpy as np

from cw.simulation import GymEnvironment


class Environment(GymEnvironment):
    def __init__(self,
                 required_states=None,
                 target_time_step=None
                 ):
        super().__init__(
            target_time_step=target_time_step,
            required_states=required_states,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(3)

    def act(self, action: Any):
        pass

    def observe(self, done: bool) -> Tuple[Any, float, dict]:
        if done:
            return np.array([self.s.h, self.s.vic[1]]), 0., {'time': self.s.t}
        else:
            return np.array([self.s.h, self.s.vic[1]]), self.s.h, {'time': self.s.t}
