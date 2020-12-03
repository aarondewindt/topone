from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pickle
import time

from cw.simulation import ModuleBase


class AgentBase(ModuleBase, ABC):
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

        metadata = self.get_metadata()
        with (self.path / f"agent.{label}.pickle").open("wb") as f:
            self.last_save_time = time.time()
            pickle.dump((metadata, self.last_save_time), f)

    def load(self, label, missing_ok=False):
        if self.path is None:
            return

        self.path.mkdir(exist_ok=True)

        agent_metadata_path = self.path / f"agent.{label}.pickle"
        if agent_metadata_path.exists():
            with agent_metadata_path.open("rb") as f:
                metadata, self.last_save_time = pickle.load(f)
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

