from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import inspect
import pickle
import time

import gym

from cw.simulation import GymEnvironment


class AgentBase(ABC):
    def __init__(self,
                 environment: gym.Env,
                 path: Optional[Path]=None,
                 ):
        self.path = None if path is None else Path(path)
        self.last_save_time = None
        self.last_save_idx = None
        self.next_save_idx = 0
        self.environment = environment

        if len(idxs := self.get_backup_indices()):
            self.last_save_idx = idxs[-1]
            self.next_save_idx = self.last_save_idx + 1

    def get_backup_indices(self):
        idxs = []
        for path in self.path.glob(f"agent.*.pickle"):
            label = path.suffixes[0][1:]
            try:
                idxs.append(int(label))
            except ValueError:
                pass
        return tuple(sorted(idxs))

    def save(self, label=None):
        if self.path is None:
            return

        self.path.mkdir(exist_ok=True)

        if label is None:
            idxs = self.get_backup_indices()
            label = idxs[-1] + 1 if idxs else 0
            self.last_save_idx = label
            self.next_save_idx = label + 1

        metadata = self.get_metadata()
        with (self.path / f"agent.{label}.pickle").open("wb") as f:
            self.last_save_time = time.time()
            pickle.dump((2, metadata, self.last_save_time), f)

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
                        score_sum, total_score_sum, \
                        n_episodes, n_total_episodes = data
                    elif version == 2:
                        _, metadata, self.last_save_time = data
                    else:
                        raise NotImplementedError(f"Agent datafile format version '{version}' not supported.")

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
