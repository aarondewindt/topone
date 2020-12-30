from math import pi, cos, sin, sqrt, atan2
from typing import Sequence, Optional, Tuple

from dataclasses import dataclass, fields

import numpy as np
import numba as nb
from numba.experimental import jitclass

from cw.constants import g_earth
from cw.control import PIDScalarNumba
from ..integrators import AB3Integrator


nan = np.float64(np.nan)
nan2 = np.array([nan, nan], dtype=np.float64)
nan22 = np.array([[nan, nan],
                  [nan, nan]], dtype=np.float64)

# Stage state
ST_ENGINE_OFF = 0
ST_ENGINE_ON = 1
ST_NO_IGNITIONS = 2

# Autopilot modes
AP_NONE = 0
AP_FLIGHT_PATH_CONTROL = 1
AP_PITCH_CONTROL = 2
AP_PITCH_RATE_CONTROL = 3

tau = 2 * pi
hpi = 0.5 * pi

stage_spec = nb.types.Tuple((nb.float64, nb.float64, nb.float64, nb.float64, nb.int32))

# Let numba figure out the type of this thing.
pid_scalar_numba_type = nb.typeof(PIDScalarNumba(0, 0, 0))

sim_spec = (
    ("surface_diameter", nb.float64),
    ("mu", nb.float64),
    ("stages", nb.types.ListType(stage_spec)),
    ("last_stage_idx", nb.int32),
    ("current_stage", stage_spec),
    ("current_stage_has_infinite_ignitions", nb.boolean),
    ("initial_longitude", nb.float64),
    ("initial_altitude", nb.float64),
    ("initial_theta_e", nb.float64),
    ("t", nb.float64),
    ("action_engine_on", nb.boolean),
    ("action_drop_stage", nb.boolean),
    ("action_autopilot_mode", nb.int32),
    ("action_autopilot_reference", nb.float64),
    ("ap_comm_gamma_e", nb.float64),
    ("ap_comm_theta_e", nb.float64),
    ("gii", nb.types.Array(nb.float64, 1, "C")),
    ("xii", nb.types.Array(nb.float64, 1, "C")),
    ("vii", nb.types.Array(nb.float64, 1, "C")),
    ("aii", nb.types.Array(nb.float64, 1, "C")),
    ("tei", nb.types.Array(nb.float64, 2, "C")),
    ("vie", nb.types.Array(nb.float64, 1, "C")),
    ("fii_thrust", nb.types.Array(nb.float64, 1, "C")),
    ("theta_i", nb.float64),
    ("theta_i_dot", nb.float64),
    ("theta_e", nb.float64),
    ("mass", nb.float64),
    ("mass_dot", nb.float64),
    ("h", nb.float64),
    ("engine_on", nb.boolean),
    ("stage_state", nb.int32),
    ("stage_idx", nb.int32),
    ("stage_ignitions_left", nb.int32),
    ("gamma_i", nb.float64),
    ("gamma_e", nb.float64),
    ("longitude", nb.float64),
    ("reward", nb.float64),
    ("score", nb.float64),
    ("done", nb.boolean),

    # There is no way to know the numba type of these classes, so instead
    # we create a new instance and let numba figure out the type.
    ("integrator", nb.typeof(AB3Integrator(0.01, 7))),
    ("gamma_controller", pid_scalar_numba_type),
    ("theta_controller", pid_scalar_numba_type)
)


@jitclass(sim_spec)
class Simulation:
    def __init__(self,
                 dt: float,
                 surface_diameter: float,
                 mu: float,
                 stages: Sequence[Tuple[float, float, float, float, int]],
                 initial_longitude: float,
                 initial_altitude: float,
                 initial_theta_e: float,
                 gamma_controller_gains: Tuple[float, float, float],
                 theta_controller_gains: Tuple[float, float, float]):

        self.integrator = AB3Integrator(dt, 7)

        self.gamma_controller = PIDScalarNumba(*gamma_controller_gains)
        self.theta_controller = PIDScalarNumba(*theta_controller_gains)

        self.surface_diameter = surface_diameter
        self.mu = mu
        self.stages = stages
        self.last_stage_idx = len(stages) - 1
        self.current_stage = self.stages[0]
        self.current_stage_has_infinite_ignitions = False

        self.initial_longitude = initial_longitude
        self.initial_altitude = initial_altitude
        self.initial_theta_e = initial_theta_e

        self.t: float = nan
        self.action_engine_on: bool = False
        self.action_drop_stage: bool = False
        self.action_autopilot_mode: int = 0
        self.action_autopilot_reference: float = 0.

        self.ap_comm_gamma_e: float = nan
        self.ap_comm_theta_e: float = nan

        self.gii: np.ndarray = nan2
        self.xii: np.ndarray = nan2
        self.vii: np.ndarray = nan2
        self.aii: np.ndarray = nan2
        self.tei: np.ndarray = nan22
        self.vie: np.ndarray = nan2
        self.fii_thrust: np.ndarray = nan2
        self.theta_i: float = nan
        self.theta_i_dot: float = nan
        self.theta_e: float = nan
        self.mass: float = nan
        self.mass_dot: float = nan
        self.h: float = nan
        self.engine_on: bool = False
        self.stage_state: int = 0
        self.stage_idx: int = 0
        self.stage_ignitions_left: int = 0
        self.gamma_i: float = nan
        self.gamma_e: float = nan
        self.longitude: float = nan

        self.reward: float = nan
        self.score: float = nan
        self.done: bool = False

        self.reset()

    def get_y_dot(self):
        y = np.empty(7, dtype=np.float64)
        y[:2] = self.vii
        y[2:4] = self.aii
        y[4] = self.theta_i_dot
        y[5] = self.mass_dot
        y[6] = self.reward
        return y

    def get_y(self):
        y = np.empty(7, dtype=np.float64)
        y[:2] = self.xii
        y[2:4] = self.vii
        y[4] = self.theta_i
        y[5] = self.mass
        y[6] = self.score
        return y

    def set_t_y(self, t, y):
        self.t = t
        self.xii[0] = y[0]
        self.xii[1] = y[1]
        self.vii[0] = y[2]
        self.vii[1] = y[3]
        self.theta_i = y[4]
        self.mass = y[5]
        self.score = y[6]

    def reset(self):
        self.gamma_controller.reset()
        self.theta_controller.reset()

        self.t = 0.0
        self.action_engine_on: bool = False
        self.action_drop_stage: bool = False
        self.action_autopilot_mode: int = 0
        self.action_autopilot_reference: float = 0.0

        self.ap_comm_gamma_e: float = np.nan
        self.ap_comm_theta_e: float = np.nan

        self.current_stage = self.stages[0]
        self.current_stage_has_infinite_ignitions = self.current_stage[4] < 0

        # current_stage.dry_mass + current_stage.propellant_mass
        self.mass = self.current_stage[0] + self.current_stage[1]
        self.stage_idx = 0
        self.stage_state = ST_ENGINE_OFF

        self.theta_e = self.initial_theta_e
        self.theta_i = -hpi + self.initial_longitude + self.initial_theta_e
        self.theta_i_dot = 0.0

        initial_r = self.initial_altitude + self.surface_diameter
        self.vii = np.array((0, 0), dtype=np.float64)
        self.xii = np.array([
            cos(self.initial_longitude) * initial_r,
            sin(self.initial_longitude) * initial_r
        ])

        self.tei = np.array(((-sin(self.initial_longitude), cos(self.initial_longitude)),
                             (cos(self.initial_longitude), sin(self.initial_longitude))), dtype=np.float64)

        self.reward = 0
        self.score = 0

        self._update_states()
        y = self.get_y()
        self.integrator.reset(y)

    def step(self, action):
        self.action_engine_on, \
            self.action_drop_stage,\
            self.action_autopilot_mode,\
            self.action_autopilot_reference = action

        t, y = self.integrator.step(self.get_y_dot())
        self.set_t_y(t, y)

        self._update_states()

    def _update_states(self):
        # The old code makes use of s. I decided to keep it
        # so I can distiguish states with constants (self attributes)
        s = self

        r = sqrt(s.xii[0] ** 2 + s.xii[1] ** 2)
        s.h = r - self.surface_diameter
        s.gii = -self.mu * s.xii / (r ** 3)

        s.gamma_i = atan2(s.vii[1], s.vii[0])
        s.longitude = atan2(s.xii[1], s.xii[0])
        s.gamma_e = hpi + s.gamma_i - s.longitude

        s.theta_e = hpi + s.theta_i - s.longitude

        if self.action_autopilot_mode == AP_NONE:
            self.ap_comm_gamma_e = np.nan
            self.ap_comm_theta_e = np.nan
            self.gamma_controller.reset()
            self.theta_controller.reset()
        elif self.action_autopilot_mode == AP_FLIGHT_PATH_CONTROL:
            self.ap_comm_gamma_e = self.action_autopilot_reference
            self.ap_comm_theta_e = s.theta_e + self.gamma_controller.step(s.t, s.ap_comm_gamma_e, s.gamma_e)
            self.theta_i_dot = self.theta_controller.step(s.t, s.ap_comm_theta_e, s.theta_e)
        elif self.action_autopilot_mode == AP_PITCH_CONTROL:
            self.ap_comm_gamma_e = np.nan
            self.gamma_controller.reset()
            self.ap_comm_theta_e = self.action_autopilot_reference
            self.theta_i_dot = self.theta_controller.step(s.t, s.ap_comm_theta_e, s.theta_e)
        elif self.action_autopilot_mode == AP_PITCH_RATE_CONTROL:
            self.ap_comm_gamma_e = np.nan
            self.ap_comm_theta_e = np.nan
            self.gamma_controller.reset()
            self.theta_controller.reset()
            self.theta_i_dot = self.action_autopilot_reference

        s.tei[0, 0] = -sin(s.longitude)
        s.tei[0, 1] = cos(s.longitude)
        s.tei[1, 0] = cos(s.longitude)
        s.tei[1, 1] = sin(s.longitude)

        s.vie = s.tei @ s.vii

        # Keep the engine off if it's been fired already.
        if s.stage_state == ST_NO_IGNITIONS:
            s.engine_on = False

        # Go to the firing state if we get the action to do so.
        # Remove one ignition
        elif s.stage_state == ST_ENGINE_OFF:
            if s.action_engine_on:
                if not self.current_stage_has_infinite_ignitions:
                    s.stage_ignitions_left -= 1
                s.stage_state = ST_ENGINE_ON
                s.engine_on = True

        # Turn the engine off and go to the fired state if the engine has been
        # turned off or ran out of propellant.
        else:
            if not s.action_engine_on:
                s.engine_on = False
                if (s.stage_ignitions_left == 0) or (s.mass <= self.current_stage[0]):
                    s.stage_state = ST_NO_IGNITIONS
                else:
                    s.stage_state = ST_ENGINE_OFF

        if s.engine_on:
            # 3: thrust, 2: specific impulse
            s.mass_dot = -self.current_stage[3] / self.current_stage[2] / g_earth
            s.fii_thrust = np.array([cos(s.theta_i) * self.current_stage[3],
                                     sin(s.theta_i) * self.current_stage[3]])
        else:
            s.mass_dot = 0
            s.fii_thrust = np.zeros(2, dtype=np.float64)

        if s.action_drop_stage:
            s.action_drop_stage = False
            if s.stage_idx < self.last_stage_idx:
                # Switch to the next state.
                s.stage_idx += 1
                self.current_stage = self.stages[s.stage_idx]
                s.mass = self.current_stage[0] + self.current_stage[1]
                s.stage_state = ST_ENGINE_OFF

                # If we have infinite ignitions, set the number of ignitions left to -1.
                if self.current_stage[4] < 0:
                    s.stage_ignitions_left = 1
                    self.current_stage_has_infinite_ignitions = True
                else:
                    self.current_stage_has_infinite_ignitions = False
                    s.stage_ignitions_left = self.current_stage[4]

        s.aii = s.gii + s.fii_thrust / s.mass
        # print(f"{s.aii=}")

