from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import time

from cw.simulation import ModuleBase


class AgentBase(ModuleBase, ABC):
    def __init__(self,
                 path: Path,
                 target_time_step: float=0.1,
                 required_states=None
                 ):
        super().__init__(
            target_time_step=target_time_step,
            is_discreet=True,
            required_states=required_states
        )
        self.path = Path(path)
        self.last_save_time = None

    def save(self, label=None):
        self.path.mkdir(exist_ok=True)

        if label is None:
            idxs = []
            for path in self.path.glob(f"agent.*.pickle"):
                label = path.suffixes[0][1:]
                try:
                    idxs.append(int(label))
                except ValueError:
                    pass

            label = max(idxs) if idxs else 0

        metadata = self.get_metadata()
        with (self.path / f"agent.{label}.pickle").open("wb") as f:
            self.last_save_time = time.time()
            pickle.dump((metadata, self.last_save_time), f)

    def load(self, label, missing_ok=False):
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
        idxs = []
        for path in self.path.glob(f"agent.*.pickle"):
            label = path.suffixes[0][1:]
            try:
                idxs.append(int(label))
            except ValueError:
                pass

        if idxs:
            self.load(max(idxs), missing_ok)
        else:
            if not missing_ok:
                raise FileNotFoundError("No stored agent data found.")

    def clean(self, except_last=True):
        idxs = []
        for path in self.path.glob(f"agent.*.pickle"):
            label = path.suffixes[0][1:]
            try:
                idxs.append(int(label))
            except ValueError:
                pass
        last_idx = max(idxs)
        for idx in idxs:
            if except_last and (idx == last_idx):
                continue
            (self.path / f"agent.{idx}.pickle").unlink(missing_ok=True)

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def set_metadata(self, metadata):
        pass

