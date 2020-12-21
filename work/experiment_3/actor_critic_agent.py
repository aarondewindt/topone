import random
from collections import namedtuple
from typing import Optional, Tuple, List, cast
from pathlib import Path

import numpy as np
import tensorflow as tf
import semver
from tensorflow.keras import layers
from tqdm.auto import trange

import gym

from cw.simulation import GymEnvironment

from topone.agent_base import AgentBase
from topone.environment_base import EnvironmentBase


State = namedtuple("State", ("stage_state",))

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


class ActorCriticAgent(AgentBase):
    def __init__(self, *,
                 alpha: float,
                 gamma: float,
                 environment: GymEnvironment,
                 path: Path=None,
                 load_last=True):
        super().__init__(
            environment=environment,
            path=path,
        )
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

        self.model: tf.keras.Model = None

        self.critic_loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        if load_last:
            self.load_last(missing_ok=True)

        if self.model is None:
            self.model = ActorCriticNetwork.create_new(self.environment.action_space.n, 32)

    def get_metadata(self):
        return {
            "version": "0.0.0",
            "gamma": self.gamma,
            "model_config": self.model.get_config(),
            "optimizer": self.optimizer.get_config(),
            "critic_loss_function": self.critic_loss_function.get_config()
        }

    def set_metadata(self, metadata):
        version = semver.parse_version_info(metadata["version"])
        if version == "0.0.0":
            self.gamma = metadata["gamma"]
            self.model = ActorCriticNetwork.from_config(metadata["model_config"])
            self.optimizer = tf.keras.optimizers.Adam.from_config(metadata['optimizer'])
            self.critic_loss_function = tf.keras.losses.Huber.from_config(metadata['critic_loss_function'])
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

    def run_episode(self, initial_state: tf.Tensor, n_max_steps):
        if self.environment.thread_running:
            action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

            initial_state_shape = initial_state.shape
            state = initial_state

            for i in tf.range(n_max_steps):
                # Convert state into a batched tensor (batch size = 1)
                state = tf.expand_dims(state, 0)

                # Run the model and to get action probabilities and critic value
                action_logits_t, value = self.model(state)

                # Sample next action from the action probability distribution
                action = tf.random.categorical(action_logits_t, 1)[0, 0]
                action_probs_t = tf.nn.softmax(action_logits_t)

                # Store critic values
                values = values.write(i, tf.squeeze(value))

                # Store log probability of the action chosen
                action_probs = action_probs.write(i, action_probs_t[0, action])

                # Apply action to the environment to get next state and reward
                state, reward, done = self.tf_env_step(action)
                state.set_shape(initial_state_shape)

                # Store reward
                rewards = rewards.write(i, reward)

                if tf.cast(done, tf.bool):
                    break

            action_probs = action_probs.stack()
            values = values.stack()
            rewards = rewards.stack()

            return action_probs, values, rewards
        else:
            raise Exception("Batch not running.")

    def get_expected_return(self,
                            rewards: tf.Tensor,
                            standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + eps))

        return returns

    def compute_loss(self,
                     action_probs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.critic_loss_function(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(self,
                   initial_state: tf.Tensor,
                   model: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   n_max_steps) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(
                initial_state, n_max_steps)

            # Calculate expected returns
            returns = self.get_expected_return(rewards)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

    def train(self, n_max_steps, n_episodes):
        running_reward = 0

        with trange(n_episodes) as p:
            for episode_idx in p:
                initial_state = tf.constant(self.environment.reset(), dtype=tf.float32)
                episode_reward = int(self.train_step(initial_state, self.model, self.optimizer, n_max_steps+2))

                running_reward = episode_reward * 0.01 + running_reward * .99

                p.set_description(f'Episode {episode_idx}')
                p.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

    def run_episode_greedy(self, n_max_steps):
        self.environment.simulation.logging.log_full_episode = True
        state = tf.constant(self.environment.reset(), dtype=tf.float32)
        future = self.environment.create_result_future()
        for i in range(1, n_max_steps + 1):
            state = tf.expand_dims(state, 0)
            action_probs, _ = self.model(state)
            action = np.argmax(np.squeeze(action_probs))

            state, _, done, _ = self.environment.step(action)
            state = tf.constant(state, dtype=tf.float32)

            if done:
                break
        result = future.result(30)
        self.environment.simulation.logging.log_full_episode = False
        return result


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
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

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
