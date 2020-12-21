import random
from collections import namedtuple
from typing import Optional
from pathlib import Path

import numpy as np

from topone.agent_module_base import AgentModuleBase
from topone.environment_base import EnvironmentBase


State = namedtuple("State", ("stage_state",))


class LinearSoftmaxAgent(AgentModuleBase):
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

        self.state_size = None
        self.action_size = None
        self.theta = None

        self.environment = None
        self.action_space = None

        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []

        self.previous_iteration = None

        if load_last:
            self.load_last(missing_ok=True)

    def initialize(self, simulation):
        super().initialize(simulation)

        self.environment: EnvironmentBase = simulation.find_modules_by_type(EnvironmentBase)[0]
        self.action_space = self.environment.action_space

        if self.theta is None:
            self.state_size = 3
            self.action_size = self.action_space.n
            self.theta = np.random.random(self.state_size * self.action_size)

    def get_metadata(self):
        return self.alpha, self.gamma, self.state_size, self.action_size, self.theta

    def set_metadata(self, metadata):
        self.alpha, self.gamma, self.state_size, self.action_size, self.theta = metadata

    def store(self, state, action, prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)

    def _feature_vector(self, s, a):
        """
        Feature vector
        """
        s = (s.stage_state == 0,
             s.stage_state == 1,
             s.stage_state == 2)

        encoded = np.zeros([self.action_size, self.state_size])
        encoded[a] = s
        return encoded.flatten()

    def _softmax(self, s, a):
        return np.exp(self.theta.dot(self._feature_vector(s, a)) / 100)

    def pi(self, s):
        """\pi(a | s)"""
        weights = np.empty(self.action_size)
        for a in range(self.action_size):
            weights[a] = self._softmax(s, a)
        return weights / np.sum(weights)

    def act(self, state):
        probs = self.pi(state)
        a = random.choices(range(0, self.action_size), weights=probs)
        a = a[0]
        pi = probs[a]
        return (a, pi)

    def _gradient(self, s, a):
        expected = 0
        probs = self.pi(s)
        for b in range(0, self.action_size):
            expected += probs[b] * self._feature_vector(s, b)
        return self._feature_vector(s, a) - expected

    def _reward_function(self, t):
        """Reward function."""
        total = 0
        for tau in range(t, len(self.rewards)):
            total += self.gamma**(tau - t) * self.rewards[tau]
        return total

    def train(self):
        self.rewards -= np.mean(self.rewards)
        self.rewards /= np.std(self.rewards)
        for t in range(len(self.states)):
            s = self.states[t]
            a = self.actions[t]
            r = self._reward_function(t)
            grad = self._gradient(s, a)
            self.theta = self.theta + self.alpha * r * grad

        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []

    def display_greedy_policy(self):
        probabilities = self.pi(State(0))
        print(f"UNFIRED: {probabilities}")

        probabilities = self.pi(State(1))
        print(f"FIRING: {probabilities}")

        probabilities = self.pi(State(2))
        print(f"FIRED: {probabilities}")

    def step(self):
        state = State(self.s.stage_state)
        action, probabilities = self.act(state)
        reward, done = yield action
        self.store(state, action, probabilities, reward)
        if done:
            self.train()
