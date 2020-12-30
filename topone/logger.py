from collections import deque
from typing import Sequence, Tuple, List, Deque, Any

import numpy as np
import xarray as xr


class Logger:
    def __init__(self):
        self.raw_episode_log: Deque[Any] = deque()
        self.raw_batch_log: Deque[Any] = deque()
        self.registry: List[Tuple[object, str, Sequence[str]]] = []

        self.time_attribute_object = None
        self.time_attribute = None

        self.episode_reset()

    def register(self, obj: object, prefix: str, attributes: Sequence[str]):
        for attribute in attributes:
            if not hasattr(obj, attribute):
                raise AttributeError(f"Object has no attribute `{attribute}`.")

        self.episode_reset()
        self.batch_reset()

        self.registry.append((obj, prefix, attributes))

    def register_time_attribute(self, obj: object, attribute: str):
        if not hasattr(obj, attribute):
            raise AttributeError(f"Object has no attribute `{attribute}`.")

        self.episode_reset()
        self.batch_reset()

        self.time_attribute_object = obj
        self.time_attribute = attribute

    def step(self):
        self.raw_episode_log.append(self.step_log())

    def step_log(self):
        step_log = [getattr(self.time_attribute_object, self.time_attribute)]
        for obj, _, attributes in self.registry:
            object_log = deque()
            step_log.append(object_log)
            for attribute in attributes:
                object_log.append(getattr(obj, attribute))
        return step_log

    def episode_reset(self):
        self.raw_episode_log = deque()

    def episode_finish(self):
        if self.raw_episode_log:
            self.raw_batch_log.append(self.raw_episode_log[-1])
        else:
            self.raw_batch_log.append(self.step_log())

        result = self.log_to_dataset(self.raw_episode_log, True)
        self.episode_reset()
        return result

    def batch_reset(self):
        self.raw_episode_log = deque()
        self.raw_batch_log = deque()

    def batch_finish(self):
        result = self.log_to_dataset(self.raw_batch_log, False)
        self.batch_reset()
        return result

    def log_to_dataset(self, log: Deque[Any], is_episode_log):
        if is_episode_log:
            dim_1_name = "t"
            dim_1_values = [step_log[0] for step_log in log]
        else:
            dim_1_name = "episode_idx"
            dim_1_values = list(range(len(log)))
        data_variables = {}

        for obj_idx, (obj, prefix, attributes) in enumerate(self.registry):
            for step_log in log:
                for value, attribute in zip(step_log[obj_idx + 1], attributes):
                    if f"{prefix}_{attribute}" not in data_variables:
                        dims = [dim_1_name]
                        if not np.isscalar(value):
                            dims.extend([f"d_{value.shape[i]}_{i}" for i in range(np.ndim(value))])
                        data_variables[f"{prefix}_{attribute}"] = [dims, deque()]
                    data_variables[f"{prefix}_{attribute}"][1].append(value)

        for name, value in data_variables.items():
            value[1] = np.array(value[1])

        return xr.Dataset({name: tuple(value) for name, value in data_variables.items()},
                          coords={dim_1_name: dim_1_values})










