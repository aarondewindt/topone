from abc import ABC, abstractmethod
from itertools import chain
from typing import Iterable, Optional

from gym import spaces

from cw.simulation import ModuleBase


class EnvironmentBase(ModuleBase, ABC):
    def __init__(self, required_states: Optional[Iterable]=None) -> None:
        required_states = required_states or ()
        super().__init__(required_states=list(chain(("reward", "score"), required_states)))

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def act(self, action):
        pass
