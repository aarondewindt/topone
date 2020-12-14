from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import inspect
import pickle
import time

from gym import Space

from cw.simulation import ModuleBase

from topone.environment_base import EnvironmentBase


class AgentModuleBase(ModuleBase, ABC):
    def __init__(self,
                 path: Optional[Path]=None,
                 target_time_step: float=0.1,
                 required_states=None
                 ):
        super().__init__(
            target_time_step=target_time_step,
            is_discreet=True,
            required_states=required_states
        )
        self.path = None if path is None else Path(path)
        self.last_save_time = None

        self.environment: EnvironmentBase = None
        self.action_space: Space = None

        self.step_generator = None

        self.score = 0
        self.total_score_sum = 0
        self.score_sum = 0
        self.n_episodes = 0
        self.n_total_episodes = 0

    def initialize(self, simulation):
        super().initialize(simulation)

        if not inspect.isgeneratorfunction(self.step):
            raise ValueError("The agent step function must be a generator function.")

        self.environment: EnvironmentBase = simulation.find_modules_by_type(EnvironmentBase)[0]
        self.action_space = self.environment.action_space
        self.step_generator = None

    def run_step(self):
        self.s = self.simulation.states

        if self.step_generator is None:
            self.step_generator = self.step()
            self.environment.act(next(self.step_generator))
        else:
            try:
                self.step_generator.send((self.s.reward, False))
            except StopIteration:
                pass

            self.step_generator = self.step()
            self.environment.act(next(self.step_generator))
            self.score += self.s.reward

        del self.s

    def end(self):
        if self.step_generator is not None:
            try:
                self.step_generator.send((self.simulation.states.reward, True))
            except StopIteration:
                pass
        self.score_sum += self.score
        self.total_score_sum += self.score
        self.n_episodes += 1
        self.n_total_episodes += 1

    @property
    def average_score(self):
        return self.score_sum / self.n_episodes

    def get_backup_indices(self):
        idxs = []
        for path in self.path.glob(f"agent.*.pickle"):
            label = path.suffixes[0][1:]
            try:
                idxs.append(int(label))
            except ValueError:
                pass
        return tuple(sorted(idxs))

    def save(self, label=None, reset_average_score=True):
        if self.path is None:
            return

        self.path.mkdir(exist_ok=True)

        if label is None:
            idxs = self.get_backup_indices()
            label = idxs[-1] + 1 if idxs else 0

        metadata = self.get_metadata()
        with (self.path / f"agent.{label}.pickle").open("wb") as f:
            self.last_save_time = time.time()
            pickle.dump((1, metadata, self.last_save_time,
                         self.score_sum, self.total_score_sum,
                         self.n_episodes, self.n_total_episodes), f)

        if reset_average_score:
            self.score_sum = 0
            self.n_episodes = 0

    def load(self, label, missing_ok=False):
        if self.path is None:
            return

        self.path.mkdir(exist_ok=True)

        agent_metadata_path = self.path / f"agent.{label}.pickle"
        if agent_metadata_path.exists():
            with agent_metadata_path.open("rb") as f:
                data = pickle.load(f)
                # I didn't think of having different versions of the agent datafile.
                # So the first version will be the only one to contain two elements.
                if len(data) == 2:  # First format version (v0)
                    metadata, self.last_save_time = pickle.load(f)
                else:
                    version = data[0]
                    if version == 1:
                        _, metadata, self.last_save_time, \
                        self.score_sum, self.total_score_sum, \
                        self.n_episodes, self.n_total_episodes = data
                    else:
                        raise NotImplemented(f"Agent datafile format version '{version}' not supported.")

            self.set_metadata(metadata)
        else:
            if not missing_ok:
                raise FileNotFoundError("No stored agent data found.")

    def load_last(self, missing_ok=False):
        if self.path is None:
            return

        self.path.mkdir(exist_ok=True)

        idxs = self.get_backup_indices()
        if idxs:
            self.load(max(idxs), missing_ok)
        else:
            if not missing_ok:
                raise FileNotFoundError("No stored agent data found.")

    def clean(self, except_last=True):
        if self.path is None:
            return

        idxs = self.get_backup_indices()
        if except_last:
            idxs = idxs[:-1]

        for idx in idxs:
            (self.path / f"agent.{idx}.pickle").unlink(missing_ok=True)

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def set_metadata(self, metadata):
        pass

    @abstractmethod
    def display_greedy_policy(self):
        pass
