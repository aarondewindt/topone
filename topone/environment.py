from typing import Sequence, Union
from collections import namedtuple
from math import sin, cos, sqrt
from enum import Enum

import numpy as np
from numba import jit
from cw.simulation import ModuleBase
from cw.constants import g_earth


Stage = namedtuple("Stage", ("dry_mass", "propellant_mass", "specific_impulse", "thrust"))


UNFIRED = 0
FIRING = 1
FIRED = 2


class Environment(ModuleBase):
    def __init__(self,
                 surface_diameter: float,
                 mu: float,
                 stages: Sequence[Stage],
                 initial_latitude: float,
                 initial_altitude: float,
                 initial_theta: float):
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
            ]
        )
        self.surface_diameter = surface_diameter
        self.mu = mu
        self.stages = stages
        self.last_stage_idx = len(stages) - 1
        self.current_stage = self.stages[0]

        self.initial_latitude = initial_latitude
        self.initial_altitude = initial_altitude
        self.initial_theta = initial_theta

    def initialize(self, simulation):
        super().initialize(simulation)
        initial_r = self.initial_altitude + self.surface_diameter
        self.current_stage = self.stages[0]

        simulation.states.mass = self.current_stage.dry_mass + self.current_stage.propellant_mass
        simulation.states.stage_idx = 0
        simulation.states.stage_state = UNFIRED
        simulation.states.theta = self.initial_theta
        simulation.states.theta_dot = 0
        simulation.states.engine_on = False
        simulation.states.vii = np.zeros(2)
        simulation.states.xii = np.array([
            cos(self.initial_latitude) * initial_r,
            sin(self.initial_latitude) * initial_r
        ])

    def numba_step(self):
        self.s.aii, self.s.mass_dot, self.s.fii_thrust, self.s.stage_idx, \
         self.s.stage_state, self.s.mass, self.s.engine_on, self.s.gii = step_numba(
            self.s.command_engine_on, self.s.command_drop_stage,
            self.s.xii, self.s.stage_state, self.s.mass, self.s.engine_on, self.s.theta, self.s.stage_idx,
            self.stages, self.current_stage, self.mu)

    def step(self):
        r = sqrt(self.s.xii[0]**2 + self.s.xii[1]**2)
        self.s.h = r - self.surface_diameter
        self.s.gii = -self.mu * self.s.xii / (r**3)

        # Switch stages.
        # Keep the engine off if it's been fired already.
        if self.s.stage_state == FIRED:
            self.s.engine_on = False

        # Go to the firing state if the engine.
        elif self.s.stage_state == UNFIRED:
            if self.s.command_engine_on:
                self.s.stage_state = FIRING
                self.s.engine_on = True

        # Turn the engine off and go to the fired state if the engine has been
        # turned off or ran out of propellant.
        else:
            if (not self.s.command_engine_on) or (self.s.mass <= self.current_stage.dry_mass):
                self.s.stage_state = FIRED
                self.s.engine_on = False

        if self.s.engine_on:
            self.s.mass_dot = -self.current_stage.thrust / self.current_stage.specific_impulse / g_earth
            self.s.fii_thrust = np.array([cos(self.s.theta) * self.current_stage.thrust,
                                          sin(self.s.theta) * self.current_stage.thrust])
        else:
            self.s.mass_dot = 0
            self.s.fii_thrust = np.zeros(2)

        if self.s.command_drop_stage:
            self.s.command_drop_stage = False
            if self.s.stage_idx < self.last_stage_idx:
                self.s.stage_idx += 1
                self.current_stage = self.stages[self.s.stage_idx]
                self.s.mass = self.current_stage.dry_mass + self.current_stage.propellant_mass
                self.s.stage_state = False
                self.s.stage_idx = self.s.stage_idx

        self.s.aii = self.s.gii + self.s.fii_thrust / self.s.mass


@jit(nopython=True)
def step_numba(command_engine_on, command_drop_stage,
               xii, stage_state, mass, engine_on, theta, stage_idx,
               stages, current_stage, mu):
    r = sqrt(xii[0] ** 2 + xii[1] ** 2)
    gii = -mu * xii / (r ** 3)

    # Switch stages.
    # Keep the engine off if it's been fired already.
    if stage_state == FIRED:
        engine_on = False

    # Go to the firing state if the engine.
    elif stage_state == UNFIRED:
        if command_engine_on:
            stage_state = FIRING

    # Turn the engine off and go to the fired state if the engine has been
    # turned off or ran out of propellant.
    else:
        if (not command_engine_on) or (mass <= current_stage.dry_mass):
            stage_state = FIRED
            engine_on = False

    if engine_on:
        mass_dot = current_stage.thrust / current_stage.specific_impulse / g_earth
        fii_thrust = np.array([cos(theta) * current_stage.thrust,
                               sin(theta) * current_stage.thrust])
    else:
        mass_dot = 0
        fii_thrust = np.zeros(2)

    if command_drop_stage:
        command_drop_stage = False
        if stage_idx < (len(stages) - 1):
            stage_idx += 1
            mass = current_stage.dry_mass + current_stage.propellant_mass
            stage_state = UNFIRED
            stage_idx = stage_idx

    aii = gii + fii_thrust / mass

    return aii, mass_dot, fii_thrust, stage_idx, stage_state, mass, engine_on, gii
