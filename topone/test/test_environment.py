import unittest
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from cw.context import time_it
from cw.simulation import Simulation, StatesBase, AB3Integrator, ModuleBase, LastValueLogger

from topone.environment import Environment, Stage
from topone.pid_agent import PIDAgent


class TestEnvironment(unittest.TestCase):
    def test_environment(self):
        simulation = Simulation(
            states_class=States,
            integrator=AB3Integrator(
                h=0.1,
                rk4=False,
                fd_max_order=1),
            modules=[
                Environment(
                    surface_diameter=1737.4e3,
                    mu=4.9048695e12,
                    stages=(
                        Stage(1, 1, 100, 3),
                    ),
                    initial_altitude=1000,
                    initial_theta=0,
                    initial_latitude=0,
                ),
                PIDAgent(1, 0, 0)
            ],
            logging=LastValueLogger(),
            initial_state_values=None,
        )

        simulation.initialize()

        with time_it("simulation run"):
            result = simulation.run(100)
            print(result)


@dataclass
class States(StatesBase):
    t: float = 0
    command_engine_on: bool = False
    command_drop_stage: bool = False
    gii: np.ndarray = np.zeros(2)
    xii: np.ndarray = np.zeros(2)
    vii: np.ndarray = np.zeros(2)
    aii: np.ndarray = np.zeros(2)
    fii_thrust: np.ndarray = np.zeros(2)
    theta: float = 0
    theta_dot: float = 0
    mass: float = 0
    mass_dot: float = 0
    h: float = 0
    engine_on: bool = False
    stage_state: int = 0
    stage_idx: int = 0

    def get_y_dot(self):
        y = np.empty(6)
        y[:2] = self.vii
        y[2:4] = self.aii
        y[4] = self.theta_dot
        y[5] = self.mass_dot
        return y

    def get_y(self):
        y = np.empty(6)
        y[:2] = self.xii
        y[2:4] = self.vii
        y[4] = self.theta
        y[5] = self.mass
        return y

    def set_t_y(self, t, y):
        self.t = t
        self.xii = y[:2]
        self.vii = y[2:4]
        self.theta = y[4]
        self.mass = y[5]


class UnittestModule(StatesBase):
    pass
