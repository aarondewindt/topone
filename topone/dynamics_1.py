from typing import Sequence, Callable, Optional
from collections import namedtuple
from math import sin, cos, sqrt, atan2, pi
from enum import Enum

import numpy as np
from numba import jit
from cw.simulation import ModuleBase, StatesBase
from cw.constants import g_earth


Stage = namedtuple("Stage", ("dry_mass", "propellant_mass", "specific_impulse", "thrust"))

UNFIRED = 0
FIRING = 1
FIRED = 2

hpi = 0.5 * pi


class Dynamics1(ModuleBase):
    def __init__(self,
                 surface_diameter: float,
                 mu: float,
                 stages: Sequence[Stage],
                 initial_latitude: float,
                 initial_altitude: float,
                 initial_theta_e: float):
        super().__init__(
            required_states=[
                # Inputs
                "command_engine_on",
                "command_drop_stage",

                # Output
                "gii",  # Gravitational acceleration in the the inertial frame expressed in the inertial frame.
                "xii", "vii", "aii",  # Inertial position, velocity and acceleration expressed in the inertial frame.
                "theta", "theta_dot",
                "mass",
                "mass_dot",
                "fii_thrust",
                "engine_on",
                "stage_state",
                "stage_idx",
                "h",
                "gamma_i",
                "gamma_e",
                "latitude"
            ]
        )
        self.surface_diameter = surface_diameter
        self.mu = mu
        self.stages = stages
        self.last_stage_idx = len(stages) - 1
        self.current_stage = self.stages[0]

        self.initial_latitude = initial_latitude
        self.initial_altitude = initial_altitude
        self.initial_theta_e = initial_theta_e

    def initialize(self, simulation):
        super().initialize(simulation)
        initial_r = self.initial_altitude + self.surface_diameter
        self.current_stage = self.stages[0]

        simulation.states.mass = self.current_stage.dry_mass + self.current_stage.propellant_mass
        simulation.states.stage_idx = 0
        simulation.states.stage_state = UNFIRED
        simulation.states.theta = -hpi + self.initial_latitude + self.initial_theta_e
        simulation.states.theta_dot = 0
        simulation.states.engine_on = False
        simulation.states.vii = np.zeros(2)
        simulation.states.xii = np.array([
            cos(self.initial_latitude) * initial_r,
            sin(self.initial_latitude) * initial_r
        ])

    def step(self):
        s = self.s
        r = sqrt(s.xii[0]**2 + s.xii[1]**2)
        s.h = r - self.surface_diameter
        s.gii = -self.mu * s.xii / (r**3)

        s.gamma_i = atan2(s.vii[1], s.vii[0])
        s.latitude = atan2(s.xii[1], s.xii[0])
        s.gamma_e = hpi + s.gamma_i - s.latitude

        # Switch stages.
        # Keep the engine off if it's been fired already.
        if s.stage_state == FIRED:
            s.engine_on = False

        # Go to the firing state if the engine.
        elif s.stage_state == UNFIRED:
            if s.command_engine_on:
                s.stage_state = FIRING
                s.engine_on = True

        # Turn the engine off and go to the fired state if the engine has been
        # turned off or ran out of propellant.
        else:
            if (not s.command_engine_on) or (s.mass <= self.current_stage.dry_mass):
                s.stage_state = FIRED
                s.engine_on = False

        if s.engine_on:
            s.mass_dot = -self.current_stage.thrust / self.current_stage.specific_impulse / g_earth
            s.fii_thrust = np.array([cos(s.theta) * self.current_stage.thrust,
                                     sin(s.theta) * self.current_stage.thrust])
        else:
            s.mass_dot = 0
            s.fii_thrust = np.zeros(2)

        if s.command_drop_stage:
            s.command_drop_stage = False
            if s.stage_idx < self.last_stage_idx:
                s.stage_idx += 1
                self.current_stage = self.stages[s.stage_idx]
                s.mass = self.current_stage.dry_mass + self.current_stage.propellant_mass
                s.stage_state = UNFIRED
                self.simulation.integrator.reset(states=True)

        s.aii = s.gii + s.fii_thrust / s.mass

    def get_attributes(self):
        return {
            "dynamics1_surface_diameter": self.surface_diameter,
            "dynamics1_mu": self.mu,
            "dynamics1_stages": self.stages,
            "dynamics1_initial_latitude": self.initial_latitude,
            "dynamics1_initial_altitude": self.initial_altitude,
            "dynamics1_initial_theta_e": self.initial_theta_e,
        }