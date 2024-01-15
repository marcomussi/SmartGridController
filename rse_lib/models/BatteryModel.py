import math

import numpy as np


def v_ocv_arg(soc, numpy=False):
    """
    Returns the value of  the Open Circuit voltage
    Args:
        soc: soc value of the
        numpy: tells

    """
    if not numpy:
        assert 0 <= soc <= 1
        v_ocv = 3.43 + 0.68 * soc - 0.68 * (soc ** 2) + 0.81 * (soc ** 3) \
                - 0.31 * math.exp(-46 * soc)
    else:
        v_ocv = 3.43 + 0.68 * soc - 0.68 * (soc ** 2) + 0.81 * (soc ** 3) \
                - 0.31 * np.exp(-46 * soc)
    return v_ocv


def v_ocv_mult(soc, mult, numpy=False):
    """
    Returns the value of the Open Circuit Voltage multiplied by a factor
    Args:
        soc: sov level
        mult: multiplier
        numpy:

    Returns:

    """
    if not numpy:
        assert 0 <= soc <= 1
    v_ocv_norm = v_ocv_arg(soc, numpy=numpy)
    return v_ocv_norm * mult


def compute_next_soc(curr_soc, current, soh, t_sample, nom_cap):
    assert 0 <= curr_soc <= 1
    assert 0 <= soh <= 1
    assert t_sample > 0
    assert nom_cap > 0
    return curr_soc - current * t_sample / (nom_cap * 3600 * soh)


class BatteryModel:
    """
    Keeps track of the battery state
    """

    def __init__(self, nominal_capacity, init_soc, starting_temp, nominal_voltage):
        """
        Constructor of a battery
        :param nominal_capacity: Expressed in Ah
        :param init_soc: [0,1]
        :param starting_temp: °C
        :param nominal_voltage: V
        """
        assert 0 <= init_soc <= 1
        self.nominal_capacity = nominal_capacity
        self.SoH = 1
        self.soc = init_soc
        self.temp = starting_temp
        self.nominal_voltage = nominal_voltage

    # this function maximum is  4.24
    def v_ocv(self):
        v_ocv = 3.43 + 0.68 * self.soc - 0.68 * (self.soc ** 2) + 0.81 * (self.soc ** 3) \
                - 0.31 * math.exp(-46 * self.soc)
        return v_ocv * self.nominal_voltage / v_ocv_arg(1)

    def get_soc(self):
        return self.soc

    def step_soc(self, current, t_sample):
        # [1] - [A] * [s] / [A*h] / [1]
        return self.soc - current * t_sample / (self.nominal_capacity * 3600 * self.SoH)

    def get_temp(self):
        return self.temp

    def get_soh(self):
        return self.SoH

    def get_nominal_capacity(self):
        return self.nominal_capacity

    def get_energy_capacity(self):
        return self.nominal_capacity * self.nominal_voltage

    def update_battery_state(self, next_soc, next_soh, next_temp):
        self.soc = next_soc
        self.SoH = next_soh
        self.temp = next_temp

    def get_c_rate(self, current):
        return current / self.nominal_capacity

    def get_nominal_voltage(self):
        return self.nominal_voltage
