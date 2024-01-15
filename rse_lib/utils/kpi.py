from abc import ABC, abstractmethod

import numpy as np
from mushroom_rl.utils import dataset

from rse_lib.models.BatteryModel import v_ocv_mult, v_ocv_arg
from rse_lib.utils.reward import reward


class AbstractKPI(ABC):

    @abstractmethod
    def get_kpi_value(self, data_set, env, **params):
        pass

    @abstractmethod
    def copy(self):
        pass


class GridBatteryKPI(AbstractKPI):

    def __init__(self, buy_price, sell_price, batt_cost, soh_limit):
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.batt_cost = batt_cost
        self.soh_limit = soh_limit

    def get_kpi_value(self, data_set, env, **params):
        if "soh_values" not in params:
            raise NameError("Not attributed called soh_values in the additional parameters")
        gym_env = env.env
        states, actions, rewards, next_states, absorbings, lasts = dataset.parse_dataset(data_set)
        soh_values = params["soh_values"]
        delta_soh = soh_values[0] - soh_values[-1]
        assert delta_soh <= 1 - self.soh_limit
        # print("")
        # print("SOMMA DELLE REWARDS = {}".format(np.sum(rewards)))
        # print("DELTA_SOH = {}, BUY_PRICE = {} SELL_PRICE = {} BATT_COST={}, SOH_LIMIT={}".format(
        #     delta_soh, self.buy_price, self.sell_price, self.batt_cost, self.soh_limit
        # ))
        r_1 = 0

        grid_energy_history = gym_env.grid_energy_history[0:gym_env.iteration]
        # r = vect_reward_kpi(buy_price=self.buy_price, sell_price=self.sell_price, batt_cost=self.batt_cost,
        #                     delta_soh=delta_soh, grid_energy_arr=grid_energy_history, soh_limit=self.soh_limit)
        n_samples = len(data_set)
        for i in range(len(soh_values)-1):
            curr_soh = soh_values[i]
            next_soh = soh_values[i+1]
            delta_soh = curr_soh - next_soh
            grid_energy = grid_energy_history[i]
            #TODO rename this reward to economic profit, its misleading, its not the reward
            r_1 += reward(buy_price=self.buy_price, sell_price=self.sell_price, battery_cost=self.batt_cost,
                          grid_energy=grid_energy, delta_soh=delta_soh, soh_limit=self.soh_limit)
        return r_1

    def copy(self):
        return GridBatteryKPI(buy_price=self.buy_price, sell_price=self.sell_price,
                              batt_cost=self.batt_cost, soh_limit=self.soh_limit)


class J_KPI(AbstractKPI):

    def __init__(self):
        pass

    def get_kpi_value(self, data_set, env, **params):
        gamma = env.info.gamma
        j = dataset.compute_J(data_set, gamma)
        return np.mean(j)

    def copy(self):
        return J_KPI()


class BatteryCostKPI(AbstractKPI):

    def __init__(self, batt_cost, soh_limit):
        self.batt_cost = batt_cost
        self.soh_limit = soh_limit

    def get_kpi_value(self, data_set, env, **params):
        if "soh_values" not in params:
            raise NameError("Not attributed called soh_values in the additional parameters")
        gym_env = env.env
        states, actions, rewards, next_states, absorbings, lasts = dataset.parse_dataset(data_set)
        soh_values = params["soh_values"]
        delta_soh = soh_values[0] - soh_values[-1]
        assert delta_soh <= 1 - self.soh_limit

        grid_energy_history = gym_env.grid_energy_history[0:gym_env.iteration]
        return vect_reward_kpi(0, 0, self.batt_cost,
                               delta_soh, grid_energy_history, self.soh_limit)

    def copy(self):
        return BatteryCostKPI(batt_cost=self.batt_cost, soh_limit=self.soh_limit)


class EnergyCostKPI(AbstractKPI):

    def __init__(self, buy_price, sell_price):
        self.buy_price = buy_price
        self.sell_price = sell_price

    def get_kpi_value(self, data_set, env, **params):
        gym_env = env.env
        end_history = len(data_set)
        grid_energy_history = gym_env.grid_energy_history[0:end_history]
        return vect_reward_kpi(self.buy_price, self.sell_price, 0,
                               0, grid_energy_history, 0)

    def copy(self):
        return EnergyCostKPI(buy_price=self.buy_price, sell_price=self.sell_price)


class DeltaSohKPI(AbstractKPI):

    def __init__(self, soh_limit):
        self.soh_limit = soh_limit

    def get_kpi_value(self, data_set, env, **params):
        soh_values = params["soh_values"]
        delta_soh = soh_values[0] - soh_values[-1]
        assert delta_soh <= 1 - self.soh_limit
        return delta_soh

    def copy(self):
        return DeltaSohKPI(soh_limit=self.soh_limit)


def vect_reward_kpi(buy_price, sell_price, batt_cost, delta_soh, grid_energy_arr, soh_limit):
    assert 0 <= delta_soh <= 1
    assert delta_soh <= 1 - soh_limit
    r = 0
    neg_grid_energy = np.zeros(grid_energy_arr.size)
    pos_grid_energy = np.zeros(grid_energy_arr.size)
    # select all negative energy, while keeping the shape of original energy array
    neg_grid_energy[grid_energy_arr < 0] = grid_energy_arr[grid_energy_arr < 0]
    # select all positive energy, while keeping the shape of original energy array
    pos_grid_energy[grid_energy_arr >= 0] = grid_energy_arr[grid_energy_arr >= 0]
    # if positive, buy, so loose money
    r -= np.sum(buy_price * np.abs(pos_grid_energy))
    # if negative, selling, so gain money
    r += np.sum(sell_price * np.abs(neg_grid_energy))
    # SoH is always a cost, therefore loose monet
    r -= batt_cost * delta_soh / (1 - soh_limit)
    return r


def reward_kpi_old(data_set, buy_price, sell_price, batt_cost, battery, t_sample, soh_limit):
    """
    This kpi returns the value of the reward ignoring eventual punishment quantities.
    cumulative_reward = sell_price * total_energy_sold - buy_price * total_energy_bought
    - total_degradation * battery_cost
    This implementation simulates the whole profile.
    :param data_set: contains the current state, the reward, the next state and the condition
    :param buy_price: price of the energy when is bought
    :param sell_price: price of the energy when is sold
    :param batt_cost: cost of the battery, depends on the capacity of the battery
    :param battery: battery model, used to retrieve some dats
    :param t_sample: sample period of the controller
    :soh_limit: lowest amount of soh admissible, id: 0.8.
    :return:
    """
    max_c_rate_index = 5
    soc_index = 0
    soh_index = 4
    m = battery.nominal_voltage / v_ocv_arg(1)
    cumulative_reward = 0
    for row in data_set:
        curr_state = row[0]
        action = row[1][0]
        r = row[2]
        next_state = row[3]

        curr_soc = curr_state[soc_index]
        curr_soh = curr_state[soh_index]
        next_soh = next_state[soh_index]
        delta_soh = abs(next_soh - curr_soh)
        max_c_rate = curr_state[max_c_rate_index]
        net_power = battery.get_nominal_capacity() * max_c_rate * v_ocv_mult(curr_soc, m)
        grid_power = (1 - action) * net_power
        grid_energy = grid_power * t_sample
        cumulative_reward += reward(buy_price=buy_price, sell_price=sell_price, battery_cost=batt_cost,
                                    grid_energy=grid_energy, delta_soh=delta_soh, soh_limit=soh_limit)

    return cumulative_reward


def optimized_reward_kpi(data_set, buy_price, sell_price, batt_cost, battery, t_sample, soh_limit):
    """
    This kpi returns the value of the reward ignoring eventual punishment quantities.
    cumulative_reward = sell_price * total_energy_sold - buy_price * total_energy_bought
    - total_degradation * battery_cost
    This implementation contains vectorization optimizations
    :param data_set: contains the current state, the reward, the next state and the condition
    :param buy_price: price of the energy when is bought
    :param sell_price: price of the energy when is sold
    :param batt_cost: cost of the battery, depends on the capacity of the battery
    :param battery: battery model, used to retrieve some dats
    :param t_sample: sample period of the controller
    :soh_limit: lowest amount of soh admissible, id: 0.8.
    :return:
    """
    m = battery.nominal_voltage / v_ocv_arg(1)

    max_c_rate_index = 5
    soc_index = 0
    soh_index = 4
    data_set_length = len(data_set)

    data_set = np.array(data_set)
    # curr state
    state_size = len(data_set[0][0])
    curr_state = np.concatenate(data_set[:, 0]).reshape(data_set_length, state_size)
    # action
    action_size = len(data_set[0][1])
    action = np.concatenate(data_set[:, 1]).reshape(data_set_length, action_size)
    # next state
    next_state = np.concatenate(data_set[:, 3]).reshape(data_set_length, state_size)
    curr_soc = curr_state[:, soc_index]
    curr_soh = curr_state[:, soh_index]
    next_soh = next_state[:, soh_index]
    delta_soh = abs(next_soh - curr_soh)
    max_c_rate = curr_state[:, max_c_rate_index]
    net_power = battery.get_nominal_capacity() * max_c_rate * v_ocv_mult(curr_soc, m, numpy=True)
    grid_power = (1 - action) * net_power
    grid_energy = grid_power * t_sample
    cumulative_rewards = np.sum(reward(buy_price=buy_price, sell_price=sell_price, battery_cost=batt_cost, grid_energy=grid_energy, delta_soh=delta_soh, soh_limit=soh_limit))
    return cumulative_rewards
