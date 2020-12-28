import numpy as np
import numba as nb
from numba.experimental import jitclass


spec = [
    ('h', nb.float64),
    ('hdiv2', nb.float64),
    ('hdiv6', nb.float64),
    ('hdiv24', nb.float64),
    ('k', nb.float64[:, :]),
    ('y', nb.float64[:]),
    ('t', nb.float64),
    ('n_steps_since_reset', nb.int64)
]


@jitclass(spec)
class AB3Integrator:
    def __init__(self, dt, n_states):
        self.h: float = dt
        self.hdiv2: float = self.h / 2
        self.hdiv6: float = self.h / 6
        self.hdiv24: float = self.h / 24

        self.k: np.ndarray = np.zeros((4, n_states), dtype=np.float64)

        self.y = np.zeros((n_states,), dtype=np.float64)
        self.t = 0.0

        self.n_steps_since_reset = 0

    def reset(self, y0, t0=0.0):
        self.y = y0
        self.n_steps_since_reset = 0
        if t0 >= 0.0:
            self.t = t0

    def change_dt(self, dt):
        self.h: float = dt
        self.hdiv2: float = self.h / 2
        self.hdiv6: float = self.h / 6
        self.hdiv24: float = self.h / 24

        self.reset(self.y, -1)

    def step(self, y0_dot):
        # print(y0_dot)
        y0 = self.y
        t0 = self.t
        self.t = t0 + self.h

        self.k[0] = self.k[1]
        self.k[1] = self.k[2]
        self.k[2] = self.k[3]
        self.k[3] = y0_dot

        if self.n_steps_since_reset < 4:
            # Run euler integration for the first 4 steps.
            self.y = y0 + self.h * y0_dot
        else:
            self.y = y0 + self.hdiv24 * (55 * self.k[3] - 59 * self.k[2] + 37 * self.k[1] - 9 * self.k[0])

        self.n_steps_since_reset += 1

        return self.t, self.y



