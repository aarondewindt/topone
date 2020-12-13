import random
from collections import namedtuple
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from topone.agent_base import AgentBase
from topone.environment_base import EnvironmentBase


State = namedtuple("State", ("stage_state",))


class ActorCriticAgent(AgentBase):
    def __init__(self, *,
                 alpha: float,
                 gamma: float,
                 path: Path=None,
                 target_time_step=0.1,
                 load_last=True):
        super().__init__(
            path=path,
            target_time_step=target_time_step,
            required_states=[
                "stage_state",
                "reward",

                "command_engine_on",
                "delta_v"
            ],
        )

        self.alpha = alpha
        self.gamma = gamma

        self.previous_iteration = None

        self.model: tf.keras.Model = None

        if load_last:
            self.load_last(missing_ok=True)

    def initialize(self, simulation):
        super().initialize(simulation)

        if self.model is None:
            self.model = ActorCriticNetwork(self.action_space.n, 32)

        self.action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    def get_metadata(self):
        return None

    def set_metadata(self, metadata):
        pass

    def display_greedy_policy(self):
        pass

    def process_environment_return(self, environment_return):
        reward, done = environment_return
        return reward, done
    
    def step(self):
        state = State(self.s.stage_state)
        action, probabilities = self.act(state)
        reward, done = yield action
        self.store(state, action, probabilities, reward)
        if done:
            self.train()


class ActorCriticNetwork(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
          self,
          num_actions: int,
          num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x, training, mask), self.critic(x, training, mask)
