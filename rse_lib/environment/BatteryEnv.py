import random
from abc import ABC

import gym
import numpy as np
import torch
from gym import spaces

import rse_lib.environment.data_generators as dataGen
import rse_lib.models.BatteryModel as bm
import rse_lib.models.BolunModel as bolun
import rse_lib.models.ThermalModel as tm
from rse_lib.models.BoundController import BoundController
from rse_lib.utils import reward as r
from rse_lib.utils.state import BatteryStateTracker


def init_env_from_dict(dictionary, seed):
    """
    Cretaes a battery environment from a dictionary whose keys have the same name
    of the attributes of the BatterEnv class
    Returns: A BatteryEnv object following the specification written in dictionary
    """

    # DISCRETE vs CONTINOUS actions
    if "action_type" not in dictionary.keys():
        assert KeyError("The configuration file does not contain the \"action_type\" keyword")
    if "n_actions" not in dictionary.keys():
        assert KeyError("The configuration file does not contain the \"n_actions\" keyword")
    action_type = dictionary["action_type"]
    n_actions = dictionary["n_actions"]

    # used to set which variables are in the state
    state_variables = None
    if "state_variables" in dictionary.keys():
        state_variables = dictionary["state_variables"]

    env = BatteryEnv(seed=seed, action_type=action_type, n_actions=n_actions, state_variables=state_variables)
    to_ignore = ["state_variables"]
    # setting environment attributes
    for key in dictionary.keys():
        if key not in to_ignore:
            setattr(env, key, dictionary[key])

    return env


class BatteryEnv(gym.Env, ABC):
    """
    Class that stores the dynamics of the environment. A domestic environment composed by a photo-voltaic panel and a
    house are considered. The domestic environment works with an accumulation system and is connected to the electric
    grid.
    Every time step, they emit respectively the generated and consumed power. The controller can select an action value
    that expresses the percentage of net power coming from the domestic system that will have to be stored (retrieved)
    in (from) the battery. The sign of the power will decide if the battery will be charged or discharge, and the remaining
    power is directed to the electric grid, generating a monetary transaction.
    """
    metadata = {'render.modes': ['human']}
    # parameters
    INIT_SOC = 0.5
    T_ENV = 25
    INIT_CYCLES = 0
    INIT_DOD = 0
    INIT_SOH = 1
    INIT_C_RATE = 0
    GAMMA = 0.9999
    PV_PATH = "data/PV_1year/"
    LOAD_PATH = "data/load/Carico.csv"
    N_YEARS = 1
    T_SAMPLE = 60  # seconds
    NOMINAL_VOLTAGE = 48
    BUY_PRICE = 0.15 / 1000  # expressed in €/Wh. Original prices where wrt kWh
    SELL_PRICE = 0.05 / 1000  # expressed in €/Wh.
    UNIT_BATTERY_COST = 1.5  # € / Wh
    MIN_SOH = 0.6
    MAX_SOC = 1
    MIN_SOC = 0
    BATTERY_CAPACITY = 150
    # low level controller
    LOW_LVL_MIN_SOC = 0
    LOW_LVL_MAX_SOC = 1
    # access indexes
    SOC_INDEX = 0
    TEMP_INDEX = 1
    CYCLE_INDEX = 2
    DOD_INDEX = 3
    SOH_INDEX = 4
    C_RATE_INDEX = 5
    SIN_DAY_INDEX = 6
    COS_DAY_INDEX = 7
    SIN_YEAR_INDEX = 8
    COS_YEAR_INDEX = 9
    DAY_PERIOD = 60 * 60 * 24
    YEAR_PERIOD = 60 * 60 * 24 * 365
    # reward function
    DEFAULT_REWARD = "kpi"
    LINEARIZED_REWARD = "linearized"
    REWARD_TYPES = [DEFAULT_REWARD, LINEARIZED_REWARD]

    def __init__(self, seed, cycling_type="streamflow", force_soh_monotony=True, action_type="continuous",
                 n_actions=None,
                 state_variables=None):
        gym.Env.__init__(self)
        # default state structure
        if state_variables is None:
            state_variables = [BatteryStateTracker.SOC, BatteryStateTracker.TEMP, BatteryStateTracker.N_CYCLES,
                               BatteryStateTracker.DOD, BatteryStateTracker.SOH, BatteryStateTracker.C_RATE,
                               BatteryStateTracker.TIME]
        # creatin of the action space. If continuous, it's a Box between 0 and 1, otherwise is a discrete set
        self.action_type = action_type
        if self.action_type == "continuous":
            self.action_space = spaces.Box(low=0, high=1, shape=(1,))
        elif self.action_type == "discrete":
            assert n_actions is not None, "action type is discrete, n_actions should be an integer," \
                                          " but its value is {}".format(n_actions)
            assert n_actions > 0, "n_actions should be a positive integer"
            self.action_space = spaces.Discrete(n_actions)
        self.gamma = self.GAMMA

        # set default values
        self.init_soc = BatteryEnv.INIT_SOC
        self.t_env = BatteryEnv.T_ENV
        self.init_cycles = BatteryEnv.INIT_CYCLES
        self.init_dod = BatteryEnv.INIT_DOD
        self.init_soh = BatteryEnv.INIT_SOH
        self.init_c_rate = BatteryEnv.INIT_C_RATE
        self.init_sin_day = np.sin(2 * np.pi / BatteryEnv.DAY_PERIOD * 0)
        self.init_cos_day = np.cos(2 * np.pi / BatteryEnv.DAY_PERIOD * 0)
        self.init_sin_year = np.sin(2 * np.pi / BatteryEnv.YEAR_PERIOD * 0)
        self.init_cos_year = np.cos(2 * np.pi / BatteryEnv.YEAR_PERIOD * 0)
        # creation of the observation space
        self.state_tracker = BatteryStateTracker()
        self.load_state_tracker(state_variables)
        self.observation_space = self.state_tracker.get_observation_space()
        # other parameters
        self.pv_path = BatteryEnv.PV_PATH
        self.load_path = BatteryEnv.LOAD_PATH
        self.n_years = BatteryEnv.N_YEARS
        self.t_sample = BatteryEnv.T_SAMPLE
        self.nominal_voltage = BatteryEnv.NOMINAL_VOLTAGE
        self.buy_price = BatteryEnv.BUY_PRICE
        self.sell_price = BatteryEnv.SELL_PRICE
        self.unit_battery_cost = BatteryEnv.UNIT_BATTERY_COST
        self.min_soh = BatteryEnv.MIN_SOH
        self.min_soc = BatteryEnv.MIN_SOC
        self.max_soc = BatteryEnv.MAX_SOC
        self.low_level_min_soc = BatteryEnv.LOW_LVL_MIN_SOC
        self.low_lvl_max_soc = BatteryEnv.LOW_LVL_MAX_SOC
        # data loading
        self.random_start = False
        self.data_generator = dataGen.DelayedDataGenerator(self.pv_path, self.load_path, random_start=self.random_start,
                                                           seed=seed, hours_shift=0)
        """
        cycle counting algorithm selection. 3 possibilities
        -rainflow: classical rainflow algorithm
        -fastflow: low performance but efficient implementation of rainfloq
        -streamflow: efficient and more precise online version of streamflow
        """
        assert cycling_type == "streamflow" or cycling_type == "fastflow" or cycling_type == "rainflow"
        self.cycling_type = cycling_type
        self.force_soh_monotony = force_soh_monotony

        # setting of models of the battery
        self.low_level_controller = BoundController(self.low_level_min_soc, self.low_lvl_max_soc)
        self.degradation_model = None
        self.thermal_model = None
        self.battery_model = None
        self.n_samples_reset = None
        self.time_horizon = np.inf
        self.battery_capacity = BatteryEnv.BATTERY_CAPACITY
        # environment attributes
        self._state = None
        self.net_profile = None
        self.iteration = 0
        self.is_charging = False
        self.current_dod = 0
        self.changed_dir = False
        self.n_cycles = 0
        self.pv_power = 0
        self.net_power = 0
        # history attributes
        self.soc_history = None
        self.temp_history = None
        self.grid_energy_history = None
        self.real_action_history = None

        # dynamic allocation of battery
        self.is_battery_dynamic = False

        # degradation parameters, for the degradation model
        self.deg_params = None
        self.lin_deg_limit = None
        # rewards parameters
        self.reward_type = BatteryEnv.DEFAULT_REWARD
        self.seed(seed)

    def load_state_tracker(self, variables_names):
        """
        Loads the variabl names in the state tracker.
        Args:
            variables_names: list of variables that have to be tracked

        """
        init_values = []
        for name in variables_names:
            if name == BatteryStateTracker.SOC:
                init_values.append(self.init_soc)
            elif name == BatteryStateTracker.TEMP:
                init_values.append(self.t_env)
            elif name == BatteryStateTracker.N_CYCLES:
                init_values.append(self.init_cycles)
            elif name == BatteryStateTracker.DOD:
                init_values.append(self.init_dod)
            elif name == BatteryStateTracker.SOH:
                init_values.append(self.init_soh)
            elif name == BatteryStateTracker.C_RATE:
                init_values.append(self.init_c_rate)
            elif name == BatteryStateTracker.TIME:
                init_values.append(0)
            elif name == BatteryStateTracker.PV_POWER:
                init_values.append(0)
            elif name == BatteryStateTracker.NET_POWER:
                init_values.append(0)
            else:
                raise ValueError("No variable {} is present. The following variables can be registered:\n"
                                 "{}".format(name, BatteryStateTracker.POSSIBLE_VARIABLES))
        self.state_tracker.load_variables(variables_names, init_values)

    def init_state_tracker(self):
        """
        Initializes the state tracker
        Returns:

        """
        for name in BatteryStateTracker.POSSIBLE_VARIABLES:
            if name == BatteryStateTracker.SOC:
                self.state_tracker.update_if_present(name, self.init_soc)
            elif name == BatteryStateTracker.TEMP:
                self.state_tracker.update_if_present(name, self.t_env)
            elif name == BatteryStateTracker.N_CYCLES:
                self.state_tracker.update_if_present(name, self.init_cycles)
            elif name == BatteryStateTracker.DOD:
                self.state_tracker.update_if_present(name, self.init_dod)
            elif name == BatteryStateTracker.SOH:
                self.state_tracker.update_if_present(name, self.init_soh)
            elif name == BatteryStateTracker.C_RATE:
                self.state_tracker.update_if_present(name, self.init_c_rate)
            elif name == BatteryStateTracker.TIME:
                self.state_tracker.update_if_present(name, 0)
            elif name == BatteryStateTracker.PV_POWER:
                self.state_tracker.update_if_present(name, 0)
            elif name == BatteryStateTracker.NET_POWER:
                self.state_tracker.update_if_present(name, 0)
            else:
                raise ValueError("No variable {} is present. The following variables can be registered:\n"
                                 "SoC, Temp. N_cycles, DoD, SoH, C_rate, Time")

    def reset(self):
        """
        Resets the state of the environment. Creates the house and battery profiles
        Returns:

        """
        self.iteration = 0
        self.net_profile = None
        # TODO get pv profile
        self.net_profile, max_consumption = self.data_generator.create_profile(years=self.n_years,
                                                                               t_sample=self.t_sample)
        self.pv_power = self.data_generator.curr_pv_profile[0]
        self.state_tracker.update_if_present(BatteryStateTracker.PV_POWER, self.pv_power)
        # MODELS
        # consumption * hours of utilization / period of utilization.
        # max_consumption è in W, 1000 sono le expected activity hours of the battery, 365 are the day
        # in which this activity is spread
        if self.is_battery_dynamic:
            bat_energy_capacity = max_consumption * 1000 / 365  # capacity in Wh
            self.battery_capacity = bat_energy_capacity / self.nominal_voltage  # capacity in Ah
        self.battery_model = bm.BatteryModel(self.battery_capacity, self.init_soc, self.t_env, self.nominal_voltage)

        self.thermal_model = tm.ThermalModel()
        self.degradation_model = bolun.BolunModel(self.init_soc, n_samples_reset=self.n_samples_reset,
                                                  deg_params=self.deg_params)
        if self.reward_type == BatteryEnv.LINEARIZED_REWARD:
            self.lin_deg_limit = bolun.BolunModel.bisection_fd(1 - self.min_soh)
        # state initialization
        self.init_state_tracker()
        max_c_rate = self.net_profile[0] / self.battery_model.v_ocv()
        self.state_tracker.update_if_present(BatteryStateTracker.C_RATE, max_c_rate)
        self._state = self.state_tracker.get_state()
        # history array supports
        self.soc_history = np.ones(len(self.net_profile) + 1, dtype=np.float)
        self.soc_history[0] = self.init_soc
        self.temp_history = np.ones(len(self.net_profile) + 1, dtype=np.float) * self.t_env
        self.temp_history[0] = self.t_env
        self.grid_energy_history = np.zeros(len(self.net_profile), dtype=np.float)
        self.real_action_history = np.zeros(len(self.net_profile) + 1, dtype=np.float)
        # dod attributes
        self.is_charging = False
        self.current_dod = 0
        self.changed_dir = True
        self.n_cycles = 0
        self._state = self.state_tracker.get_state()
        return self._state

    def step(self, action):
        """
        Computes a single step on the environment. The action is applied and the battery state changes accordingly.
        Then the reward signal is computed and the next state is generated
        Args:
            action: percentage of net power coming from the domestic environment used to charge or discharge
            the accumulation system

        Returns:

        """
        delta_soh, grid_power, current, alpha_max, real_action, delta_fd = self.apply_action(action)
        # conversion from power to energy W -> Wh. hours
        grid_energy = grid_power * self.t_sample / 3600
        self.grid_energy_history[self.iteration] = grid_energy
        reward = self.compute_reward(grid_energy=grid_energy, delta_soh=delta_soh, delta_fd=delta_fd)
        self.iteration += 1
        new_power = self.net_profile[self.iteration]  # just increased the index
        next_state = self.get_next_state(new_power)
        self._state = next_state
        metadata = {"alpha_max": alpha_max, "real_action": real_action}
        return next_state, reward, self.end_condition(), metadata

    def apply_action(self, action):
        """
        Real application of the action in the step action. This function computes the dynamic of the environment.
        Args:
            action: percentage of power sent to the battery

        Returns:  Difference in SoH, power directed on the grid, current of the battery, maximum action admissible,
        action actuated and linear degradation value.
        """
        alpha = action[0]
        # check on max alpha
        if self.action_type == "discrete":
            alpha = alpha / (self.action_space.n - 1)
        elif self.action_type == "continuous":
            alpha = alpha
        else:
            raise NotImplementedError()
        curr_soc = self.battery_model.get_soc()
        voltage = self.battery_model.v_ocv()
        full_power = self.net_profile[self.iteration]
        max_current = full_power / voltage
        batt_capacity = self.battery_model.get_nominal_capacity()
        soh = self.battery_model.get_soh()
        if max_current == 0:
            alpha_max = 1
        elif max_current > 0:  # the battery is discharging
            # [1] * [A * h ] * [1] / [A] / [s]. The 3600 transforms Ah in As
            alpha_max = (curr_soc - self.min_soc) * batt_capacity * 3600 * soh / (max_current * self.t_sample)
        else:  # charging
            alpha_max = (self.max_soc - curr_soc) * batt_capacity * 3600 * soh / (abs(max_current) * self.t_sample)

        if alpha_max >= 1:
            alpha_max = 1
        alpha = self.low_level_controller.check_action(alpha, action_max=alpha_max)
        self.real_action_history[self.iteration] = alpha

        # division of power between system and grid
        controlled_power = self.net_profile[self.iteration] * alpha
        grid_power = self.net_profile[self.iteration] * (1 - alpha)
        curr_soc = self.battery_model.get_soc()
        # next battery state values
        next_soc, next_temp, current = self.compute_next_values(controlled_power)

        delta_soc = abs(next_soc - curr_soc)
        assert 0 <= next_soc <= 1, "next_soc is not between 0 and 1, next_soc = {}".format(next_soc)
        assert 0 <= curr_soc <= 1, "curr_soc is not between 0 and 1, curr_soc = {}".format(curr_soc)
        assert 0 <= delta_soc <= 1, "delta_soc is not between 0 and 1, delta_soc = {}".format(delta_soc)
        self.compute_dod_n_cycles(delta_soc, current)
        curr_soh = self.battery_model.get_soh()
        curr_fd = self.degradation_model.get_fd()
        assert 0 <= curr_soh <= 1, "current SoH is not between 0 and 1, curr_soh = {}".format(curr_soh)
        t = (self.iteration + 1) * self.t_sample
        deg = self.compute_degradation(next_soc, next_temp, t, delta_soc, curr_soc)
        next_fd = self.degradation_model.get_fd()

        next_soh = 1 - deg
        if self.force_soh_monotony:
            if next_soh > curr_soh:
                next_soh = curr_soh
            if next_fd < curr_fd:
                next_fd = curr_fd
        # assert next_soh <= curr_soh, "The degradation function is not monotonic"
        if next_soh < 0:
            next_soh = 0
        # linear degradation fd limit
        if self.lin_deg_limit is not None and next_fd > self.lin_deg_limit:
            next_fd = self.lin_deg_limit
        self.battery_model.update_battery_state(next_soc, next_soh, next_temp)

        if self.force_soh_monotony:
            delta_soh = curr_soh - next_soh
            delta_fd = next_fd - curr_fd
        else:
            delta_soh = 0
        assert 0 <= delta_soh <= 1, "delta_soh is not between 0 and 1, delta_soh = {}".format(delta_soh)

        self.net_power = controlled_power

        return delta_soh, grid_power, current, alpha_max, alpha, delta_fd

    def compute_next_values(self, controlled_power):
        """
        Computes the next values of the state of the battery
        Args:
            controlled_power: power directed to the battery

        Returns: next SoC, next temperature, current of the battery

        """
        ts = self.t_sample
        voltage = self.battery_model.v_ocv()
        current = controlled_power / voltage
        old_soc = self.battery_model.get_soc()

        next_soc = self.battery_model.step_soc(current, ts)
        self.soc_history[self.iteration + 1] = next_soc
        curr_temp = self.battery_model.get_temp()
        next_temp = self.thermal_model.step_temp(current, ts, curr_temp, self.t_env)
        self.temp_history[self.iteration + 1] = next_temp
        return next_soc, next_temp, current

    def get_next_state(self, next_power):
        """
        Returns the next state that the environment will have
        Args:
            next_power: net power incoming the next stime step

        Returns:

        """
        soc = self.battery_model.get_soc()
        temp = self.battery_model.get_temp()
        n_cycles = self.n_cycles
        dod = self.current_dod
        soh = self.battery_model.get_soh()
        next_current = next_power / self.battery_model.v_ocv()
        max_c_rate = self.battery_model.get_c_rate(next_current)
        self.pv_power = self.data_generator.curr_pv_profile[self.iteration]
        t = self.iteration * self.t_sample
        net_power = self.net_power

        # update state tracker
        self.state_tracker.update_if_present(BatteryStateTracker.SOC, soc)
        self.state_tracker.update_if_present(BatteryStateTracker.TEMP, temp)
        self.state_tracker.update_if_present(BatteryStateTracker.DOD, dod)
        self.state_tracker.update_if_present(BatteryStateTracker.N_CYCLES, n_cycles)
        self.state_tracker.update_if_present(BatteryStateTracker.TIME, t)
        self.state_tracker.update_if_present(BatteryStateTracker.SOH, soh)
        self.state_tracker.update_if_present(BatteryStateTracker.C_RATE, max_c_rate)
        self.state_tracker.update_if_present(BatteryStateTracker.NET_POWER, net_power)
        self.state_tracker.update_if_present(BatteryStateTracker.PV_POWER, self.pv_power)

        next_state = self.state_tracker.get_state()
        return next_state

    def compute_degradation(self, next_soc, next_temp, t, delta_soc=None, curr_soc=None):
        """
        Computes the increase in degradation of the battery
        Args:
            next_soc: new SoC of the battery
            next_temp: new Temperature of the battery
            t: time elapsed from the beginning of the simulation
            delta_soc: difference in SoC value
            curr_soc: current SoC

        Returns:

        """

        if self.cycling_type == "fastflow":
            if self.iteration == 0:
                deg = self.degradation_model.compute_fastflow_degradation(next_soc, next_temp,
                                                                          delta_soc, t, self.iteration,
                                                                          False)
            else:
                deg = self.degradation_model.compute_fastflow_degradation(next_soc, next_temp,
                                                                          delta_soc, t, self.iteration,
                                                                          self.changed_dir)
        elif self.cycling_type == "streamflow":
            charging = next_soc > curr_soc
            deg = self.degradation_model.compute_streamflow_degradation(next_soc, next_temp, t, self.iteration,
                                                                        charging,
                                                                        self.temp_history[0:self.iteration + 1])
        elif self.cycling_type == "rainflow":
            deg = self.degradation_model.compute_rainflow_degradation(next_soc, next_temp, t, self.iteration + 1,
                                                                      self.soc_history[0:self.iteration + 1],
                                                                      self.temp_history[0:self.iteration + 1])
        else:
            raise ValueError("The cycling type is not streamflow, rainflow or fastflow.")
        return deg

    def compute_reward(self, grid_energy, delta_soh, delta_fd):
        """
        Computes the reward that the agent is subject to after performing and action. It supports 2 types of reward, one
        that amortizes the battery cost with the real degradation, one that uses the exponent of the degradation instead.
        Args:
            grid_energy: energy sold or bought from the grid
            delta_soh: different in soh in the last time step
            delta_fd: different in linear degradation in the last time step

        Returns: reward value

        """
        battery_cost = self.unit_battery_cost * self.battery_model.get_energy_capacity()
        reward = 0
        if self.reward_type == BatteryEnv.DEFAULT_REWARD:
            reward = r.reward(buy_price=self.buy_price, sell_price=self.sell_price, battery_cost=battery_cost,
                              grid_energy=grid_energy, delta_soh=delta_soh, soh_limit=self.min_soh)
        elif self.reward_type == BatteryEnv.LINEARIZED_REWARD:
            reward = r.linearized_deg_reward(buy_price=self.buy_price, sell_price=self.sell_price,
                                             battery_cost=battery_cost, grid_energy=grid_energy, delta_deg=delta_fd,
                                             deg_limit=self.lin_deg_limit)
        else:
            ValueError("{} reward type is not admissible."
                       " Legal values are {}".format(self.reward_type, BatteryEnv.REWARD_TYPES))
        return reward

    def get_state(self):
        return self._state

    def end_condition(self):
        # if the next step will be outside the range.
        # if self.iteration + 1 >= 10 or self.battery_model.get_soh() <= self.min_soh:
        if self.iteration + 1 >= len(self.net_profile) or self.battery_model.get_soh() <= self.min_soh:
            return True
        else:
            return False

    def compute_dod_n_cycles(self, delta_soc, current):
        # current is positive if DISCHARGE, negative if CHARGE (generator convention)
        if current >= 0 and self.is_battery_charging():
            self.is_charging = False
            self.current_dod = delta_soc
        elif current < 0 and self.is_battery_discharging():
            self.is_charging = True
            self.current_dod = delta_soc
        else:
            self.current_dod += delta_soc
        self.n_cycles += delta_soc

    def is_battery_charging(self):
        return self.is_charging

    def is_battery_discharging(self):
        return not self.is_charging

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def get_gamma(self):
        return self.gamma

    def get_time_horizon(self):
        return self.time_horizon

    def dump_to_dict(self):
        ret_dict = self.__dict__
        to_remove = ['action_space', 'observation_space', 'data_generator', 'degradation_model', 'thermal_model',
                     'battery_model', 'time_horizon', '_state', 'net_profile', 'iteration', 'battery_capacity',
                     'is_charging', 'current_dod', 'changed_dir', 'n_cycles', 'soc_history', 'temp_history',
                     'low_level_controller', 'starting_state']
        for key in to_remove:
            ret_dict.pop(key)
        return ret_dict

    def seed(self, seed=None):
        if seed is None:
            raise ValueError("Seed can not be none")
        self.data_generator.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def get_degradation_model(self):
        return self.degradation_model
