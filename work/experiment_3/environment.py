from typing import Any, Tuple

from gym import spaces

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
        print(f"*envi: performing action '{action}'")

    def observe(self, done: bool) -> Tuple[Any, float, dict]:
        print(f"*envi: observing")
        if done:
            return self.s.h, self.s.h, {'time': self.s.t}
        else:
            return 0, self.s.h, {'time': self.s.t}
