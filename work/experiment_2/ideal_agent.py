from topone.agent_base import AgentBase
from topone.environment_base import EnvironmentBase


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
                   f"  UNFIRED: {self.act([1, 0, 0, 1, 0])} \n" \
                   f"  FIRING: {self.act([0, 1, 0, 1, 0])}\n" \
                   f"  FIRED: {self.act([0, 0, 1, 1, 0])}\n" \
                   f"Stage 1\n" \
                   f"  UNFIRED: {self.act([1, 0, 0, 0, 1])} \n" \
                   f"  FIRING: {self.act([0, 1, 0, 0, 1])}\n" \
                   f"  FIRED: {self.act([0, 0, 1, 0, 1])}"
        print(contents)

    def step(self):
        action = self.act(self.environment.observation)
        self.environment.act(action)
