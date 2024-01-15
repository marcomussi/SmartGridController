from abc import ABC
from typing import overload

import numpy as np
from gym import spaces


class NumericVariable:
    """
    Class used to keep track of a single value. The simulation has to keep updating its value
    """

    def __init__(self, name, init_value, low, high):
        """
        Creates a named variable
        Args:
            name:
            init_value:
            low:
            high:
        """
        assert name is not None
        assert low <= high
        assert init_value >= low, "The new_value {} is smaller than lower bound {}".format(init_value, low)
        assert init_value <= high, "The new value {} is bigger than higher bound {}".format(init_value, high)
        self.name = name
        self.low = low
        self.high = high
        self.value = init_value

    def update(self, new_value):
        """
        Updates the value of the variable
        Args:
            new_value: new value of the variable. It must be between low and high

        Returns:

        """
        assert new_value >= self.low, "The new_value {} is smaller than lower bound {}".format(new_value, self.low)
        assert new_value <= self.high, "The new value {} is bigger than higher bound {}".format(new_value, self.high)
        self.value = new_value

    def get_value(self):
        return self.value

    def get_name(self):
        return self.name


class StateTracker(ABC):
    """
    Class used to keep track of some variable. It's the backbone of more complex classes.
    """

    def __init__(self):
        self.variables_list = list()
        self.variables_names = list()

    def to_dict(self):
        """
        Converts the state tracker into a dict
        Returns:

        """
        d = {}
        for i in range(len(self.variables_names)):
            name = self.variables_names[i]
            value = self.variables_list[i]
            d[name] = value
        return d

    def __setitem__(self, key, variable):
        """
        Once that a variable is registered, it cannot be registered again
        Args:
            key:
            variable:

        Returns:
        """
        name = key
        if name in self.variables_names:
            ValueError("Variable with name {} already registered")
        self.variables_names.append(name)
        self.variables_list.append(variable)

    def __getitem__(self, item):
        return self.variables_list[self.variables_names.index(item)]

    def clear(self) -> None:
        self.variables_list = list()
        self.variables_names = list()

    def copy(self):
        new_st = StateTracker()
        for i in range(len(self.variables_names)):
            name = self.variables_names[i]
            value = self.variables_list[i]
            new_st[name] = value
        return new_st

    def popitem(self):
        name = self.variables_names.pop()
        variable = self.variables_list.pop()
        return name, variable

    def keys(self):
        return self.variables_names[:]

    def values(self):
        return self.variables_list[:]

    def items(self):
        d = self.to_dict()
        return d.items()

    def __len__(self) -> int:
        return len(self.variables_names)

    def __delitem__(self, v) -> None:
        if v not in self.variables_names:
            return
        index = self.variables_names.index(v)
        self.variables_names.pop(index)
        self.variables_list.pop(index)

    def __iter__(self):
        return self.to_dict().__iter__()

    @overload
    def pop(self, key):
        value = self[key]
        self.__delitem__(key)
        return value

    @overload
    def get(self, key):
        index = self.variables_names.index(key)
        return self.variables_list[index]

    def __contains__(self, o: object) -> bool:
        return o in self.variables_list

    def update_if_present(self, key, var_value):
        if key in self.variables_names:
            index = self.variables_names.index(key)
            name = self.variables_names[index]
            self[name].update(var_value)

    def get_observation_space(self):
        lows = list()
        highs = list()
        for name in self.variables_names:
            lows.append(self[name].low)
            highs.append(self[name].high)
        lows = np.array(lows)
        highs = np.array(highs)
        observation_space = spaces.Box(low=lows, high=highs, shape=highs.shape)
        return observation_space

    def get_state(self):
        values = list()
        for name in self.variables_names:
            values.append(self[name].get_value())
        return np.array(values)

    def get_indexes(self):
        indexes = {self.variables_names[i]: i for i in range(len(self.variables_names))}
        return indexes



class BatteryStateTracker(StateTracker):
    """
    Battery environment specific implementation. It has some helper function to handle the battery state
    """
    MIN_SOC = 0
    MAX_SOC = 1

    MIN_TEMP = -np.inf
    MAX_TEMP = np.inf

    MIN_CYCLES = 0
    MAX_CYCLES = np.inf

    MIN_DOD = 0
    MAX_DOD = 1

    MIN_SOH = 0
    MAX_SOH = 1

    MIN_C_RATE = -np.inf
    MAX_C_RATE = np.inf

    MIN_TIME = 0
    MAX_TIME = np.inf

    MIN_PV_POWER = -np.inf
    MAX_PV_POWER = np.inf

    NEG_INF = -np.inf
    POS_INF = np.inf

    DAY_PERIOD = 60 * 60 * 24
    YEAR_PERIOD = 60 * 60 * 24 * 365

    SOC = "SoC"
    TEMP = "Temp"
    N_CYCLES = "N_cycles"
    DOD = "DoD"
    SOH = "SoH"
    C_RATE = "C_rate"
    TIME = "Time"
    PV_POWER = "PV_power"
    NET_POWER = "Net_power"
    POSSIBLE_VARIABLES = [SOC, TEMP, N_CYCLES, DOD, SOH, C_RATE, TIME, PV_POWER, NET_POWER]

    def __init__(self):
        super().__init__()

    def load_variables(self, name_list, init_values):
        """
        Allows to load variables from a name. This implementation is specific for our environment.
        Args:
            name_list: Names of the variable. This state tracker is case sensitive!
            init_values:

        Returns:

        """
        if not len(name_list) > 0:
            raise Exception("The name_list is empty")
        for i in range(len(name_list)):
            name = name_list[i]
            init_value = init_values[i]
            self.load_variable(name, init_value)

    def load_variable(self, name, init_value):
        if name == BatteryStateTracker.SOC:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_SOC, BatteryStateTracker.MAX_SOC)
        elif name == BatteryStateTracker.TEMP:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_TEMP, BatteryStateTracker.MAX_TEMP)
        elif name == BatteryStateTracker.N_CYCLES:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_CYCLES,
                                         BatteryStateTracker.MAX_CYCLES)
        elif name == BatteryStateTracker.DOD:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_DOD, BatteryStateTracker.MAX_DOD)
        elif name == BatteryStateTracker.SOH:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_SOH, BatteryStateTracker.MAX_SOH)
        elif name == BatteryStateTracker.C_RATE:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_C_RATE,
                                         BatteryStateTracker.MAX_C_RATE)
        elif name == BatteryStateTracker.TIME:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_TIME, BatteryStateTracker.MAX_TIME)
        elif name == BatteryStateTracker.PV_POWER:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.MIN_PV_POWER,
                                         BatteryStateTracker.MAX_PV_POWER)
        elif name == BatteryStateTracker.NET_POWER:
            self[name] = NumericVariable(name, init_value, BatteryStateTracker.NEG_INF, BatteryStateTracker.POS_INF)
        else:
            raise ValueError("No variable {} is present. The following variables can be registered:\n"
                             "{}".format(name, BatteryStateTracker.POSSIBLE_VARIABLES))

    def get_state(self):
        values = list()
        for name in self.variables_names:
            if name == BatteryStateTracker.TIME:
                t = self[name].get_value()
                sin_day = np.sin(2 * np.pi / BatteryStateTracker.DAY_PERIOD * t)
                cos_day = np.cos(2 * np.pi / BatteryStateTracker.DAY_PERIOD * t)
                sin_year = np.sin(2 * np.pi / BatteryStateTracker.YEAR_PERIOD * t)
                cos_year = np.cos(2 * np.pi / BatteryStateTracker.YEAR_PERIOD * t)
                values.extend([sin_day, cos_day, sin_year, cos_year])
            else:
                values.append(self[name].get_value())
        return np.array(values)

    def get_observation_space(self):
        lows = list()
        highs = list()
        for name in self.variables_names:
            if name != BatteryStateTracker.TIME:
                lows.append(self[name].low)
                highs.append(self[name].high)
            else:
                lows.extend([-1, -1, -1, -1])
                highs.extend([1, 1, 1, 1])
        lows = np.array(lows)
        highs = np.array(highs)
        observation_space = spaces.Box(low=lows, high=highs, shape=highs.shape)
        return observation_space

    def get_indexes(self):
        curr_index = 0
        indexes = dict()
        for var_name in self.variables_names:
            if var_name != self.TIME:
                indexes[var_name] = curr_index
                curr_index += 1
            else:
                # the time dimension is 4-dimensional
                indexes[var_name] = np.arange(curr_index, curr_index+4)
                curr_index += 4
        return indexes

