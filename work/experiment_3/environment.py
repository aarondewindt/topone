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

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(1),
            spaces.Discrete(1),
            spaces.Discrete(1),
            # spaces.Box(low=np.inf, high=np.inf, shape=(1,))
        ))

    def act(self, action: Any):
        self.s.command_engine_on = bool(action)

    def observe(self, done: bool) -> Tuple[Any, float, dict]:
        # print("done", done)
        observation = np.array((
            self.s.stage_state == 0,
            self.s.stage_state == 1,
            self.s.stage_state == 2,
            # self.s.h
        ))

        reward = self.s.h if done else 0
        info = {'time': self.s.t}

        if self.s.t > .1 and (self.s.vic[1] <= 0):
            # print("stop")
            self.simulation.stop()

        return observation, reward, info
