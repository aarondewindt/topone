import random
from collections import namedtuple
from pathlib import Path

import numpy as np

from topone.agent_base import AgentBase
from topone.environment_base import EnvironmentBase


State = namedtuple("State", ("stage_state", "stage_idx"))


class LinearSoftmaxAgent(AgentBase):
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

        self.action_size = None
        self.theta = None

        self.environment: EnvironmentBase = None
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
            self.action_size = self.action_space.n
            self.theta = np.random.random(11 * self.action_size)

    def get_metadata(self):
        return self.alpha, self.gamma, self.action_size, self.theta

    def set_metadata(self, metadata):
        self.alpha, self.gamma, self.action_size, self.theta = metadata

    def store(self, state, action, prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)

    def _x(self, s, a):
        """
        Feature vector
        """

        s = (
            s[0] == 0,
            s[0] == 1,
            s[0] == 2,
            s[1] == 0,
            s[1] == 1,
        )

        s = (
            *s,

            s[0] * s[3],
            s[0] * s[4],

            s[1] * s[3],
            s[1] * s[4],

            s[1] * s[3],
            s[1] * s[4],
        )

        encoded = np.zeros([self.action_size, 11])
        encoded[a] = s
        return encoded.flatten()

    def _softmax(self, s, a):
        return np.exp(self.theta.dot(self._x(s, a)) / 100)

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
            expected += probs[b] * self._x(s, b)
        return self._x(s, a) - expected

    def _reward_function(self, t):
        """Reward function."""
        total = 0
        for tau in range(t, len(self.rewards)):
            total += self.gamma**(tau - t) * self.rewards[tau]
        return total

    def train(self):
        if len(self.rewards) == 0:
            raise RuntimeError("No training data collected.")

        self.rewards -= np.mean(self.rewards)
        self.rewards /= np.std(self.rewards)
        for t in range(len(self.states)):
            s = self.states[t]
            a = self.actions[t]
            r = self._reward_function(t)
            grad = self._gradient(s, a)
            self.theta = self.theta + self.alpha * r * grad
        # print(self.theta)
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []

    def display_greedy_policy(self):
        contents = f"Stage 0\n" \
                   f"  UNFIRED: {np.argmax((p := self.pi(State(0, 0))))} {p} \n" \
                   f"  FIRING: {np.argmax((p := self.pi(State(1, 0))))} {p}\n" \
                   f"  FIRED: {np.argmax((p := self.pi(State(2, 0))))} {p}\n" \
                   f"Stage 1\n" \
                   f"  UNFIRED: {np.argmax((p := self.pi(State(0, 1))))} {p} \n" \
                   f"  FIRING: {np.argmax((p := self.pi(State(1, 1))))} {p}\n" \
                   f"  FIRED: {np.argmax((p := self.pi(State(2, 1))))} {p}"
        print(contents)

    def step(self):
        state = State(self.s.stage_state,self.s.stage_idx)
        action, probabilities = self.act(state)
        reward, done = yield action
        self.store(state, action, probabilities, reward)
        if done:
            self.train()
