from math import atan2

from cw.simulation import ModuleBase
from cw.control import PIDController


class PIDAgent(ModuleBase):
    def __init__(self, k_p, k_i, k_d):
        super().__init__(
            is_discreet=True,
            target_time_step=0.1,
            required_states=[
                "theta", "theta_dot"
            ],
        )

        self.controller = PIDController(k_p, k_i, k_d)

    def initialize(self, simulation):
        super().initialize(simulation)

    def step(self):
        self.controller.command = atan2(self.s.xii[1], self.s.xii[0])
        self.s.theta_dot = self.controller.step(self.s.t, self.s.theta)
        if self.s.t > 1:
            self.s.command_engine_on = True
