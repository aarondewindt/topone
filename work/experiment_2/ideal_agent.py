from collections import namedtuple

from topone.agent_base import AgentBase
from topone.environment_base import EnvironmentBase


State = namedtuple("State", ("stage_state", "stage_idx"))


class IdealAgent(AgentBase):
    def __init__(self, *,
                 target_time_step=0.1,):
        super().__init__(
            target_time_step=target_time_step,
            required_states=[
                "stage_state",
                "stage_idx",
                "reward",

                "command_engine_on",
                "command_drop_stage",
                "delta_v"
            ],
            path=None
        )
        self.environment: EnvironmentBase = None

        self.policy = [[1, 1],
                       [1, 1],
                       [2, 2]]

    def initialize(self, simulation):
        super().initialize(simulation)
        self.environment = simulation.find_modules_by_type(EnvironmentBase)[0]

    def get_metadata(self):
        return None

    def set_metadata(self, metadata):
        pass

    def act(self, state) -> int:
        if state[3]:  # Stage 1
            if state[0]:  # UNFIRED
                return 2  # Engine On
            elif state[1]:  # FIRING
                return 2  # Engine On
            elif state[2]:  # FIRED:
                return 2  # Drop stage
        else:  # Stage 2
            if state[0]:  # UNFIRED
                return 1  # Engine On
            elif state[1]:  # FIRING
                return 1  # Engine On
            elif state[2]:  # FIRED:
                return 0  # Engine Off

    def display_value(self):
        pass

    def display_greedy_policy(self):
        contents = f"Stage 0\n" \
                   f"  UNFIRED: {self.policy[0][0]} \n" \
                   f"  FIRING: {self.policy[1][0]}\n" \
                   f"  FIRED: {self.policy[2][0]}\n" \
                   f"Stage 1\n" \
                   f"  UNFIRED: {self.policy[0][1]} \n" \
                   f"  FIRING: {self.policy[1][1]}\n" \
                   f"  FIRED: {self.policy[2][1]}"
        print(contents)

    def step(self):
        action = self.policy[self.s.stage_state][self.s.stage_idx]
        reward, done = yield action
