import math
import numpy as np


def _tiny():
    return np.finfo('float64').eps


def calculate_dsr(rt, last_vt, last_wt):
    delta_vt = rt - last_vt
    delta_wt = rt**2 - last_wt
    return (last_wt * delta_vt - 0.5 * last_vt * delta_wt) / ((last_wt - last_vt**2)**(3/2) + _tiny())


class DifferentialSharpeRatio:
    def __init__(self, decay_rate=0.003):
        self.last_vt = 0
        self.last_wt = 0
        self.decay_rate = decay_rate
        self.last_sr = 0

    def update(self, rt):
        dsr = calculate_dsr(rt, self.last_vt, self.last_wt)
        self.last_vt += self.decay_rate * (rt - self.last_vt)
        self.last_wt += self.decay_rate * (rt**2 - self.last_wt)

        self.last_sr += self.decay_rate * dsr
        return dsr
