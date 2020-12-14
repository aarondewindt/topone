import random
from collections import namedtuple
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import tensorflow as tf
import semver
from tensorflow.keras import layers

import gym


from topone.agent_base import AgentBase
from topone.environment_base import EnvironmentBase


State = namedtuple("State", ("stage_state",))


class ActorCriticAgentModule(AgentBase):
    def __init__(self, *,
                 alpha: float,
                 gamma: float,
                 environment: gym.Env,
                 path: Path=None,
                 load_last=True):
        super().__init__(
            environment=environment,
            path=path,
        )

        self.alpha = alpha
        self.gamma = gamma

        self.model: tf.keras.Model = None

        if load_last:
            self.load_last(missing_ok=True)

        if self.model is None:
            self.model = ActorCriticNetwork.create_new(self.environment.action_space.n, 32)

    def run_episode(self):
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    def get_metadata(self):
        return {
            "version": "0.0.0",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "model_config": self.model.get_config(),
        }

    def set_metadata(self, metadata):
        version = semver.parse_version_info(metadata["version"])
        if version == "0.0.0":
            self.alpha = metadata["alpha"]
            self.gamma = metadata["gamma"]
            self.model = ActorCriticNetwork.from_config(metadata["model_config"])
        else:
            raise NotImplemented(f"ActorCriticAgentModule metadata version `{metadata['version']}` not supported")

    def display_greedy_policy(self):
        pass

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        state, reward, done, _ = self.environment.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])


class ActorCriticNetwork(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self,
                 common_model,
                 actor_model,
                 critic_model):
        super().__init__()
        self.common = common_model
        self.actor = actor_model
        self.critic = critic_model

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs, training, mask)
        return self.actor(x, training, mask), self.critic(x, training, mask)

    @classmethod
    def create_new(cls, num_hidden_units: int, num_actions: int):
        return cls(
            common_model=layers.Dense(num_hidden_units, activation="relu"),
            actor_model=layers.Dense(num_actions),
            critic_model=layers.Dense(1))

    def get_config(self):
        return {
            "version": "0.0.0",
            "common_config": self.common.get_config(),
            "actor_config": self.actor.get_config(),
            "critic_config": self.critic.get_config(),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        version = semver.parse_version_info(config["version"])
        if version == "0.0.0":
            return cls(
                common_model=layers.Dense.from_config(config["common_config"]),
                actor_model=layers.Dense.from_config(config["actor_config"]),
                critic_model=layers.Dense.from_config(config["critic_config"]),
            )
        else:
            raise NotImplemented(f"ActorCriticNetwork config version `{config['version']}` is not implemented.")
