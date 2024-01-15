def reward(buy_price, sell_price, battery_cost, grid_energy, delta_soh, soh_limit):
    """
    :param buy_price: price of buying energy. generally higher than sell_price. €/Wh
    :param sell_price: money gained selling energy. €/Wh
    :param grid_energy: if negative, it's the power sent from the system to the grid.
    If positive, it's the power bought from the grid used to fuel the system. Expressed in Wh
    :param delta_soh: variation of SoH of the battery
    :param battery_cost: cost of replacing the battery
    :return: reward
    :param soh_limit: lowest soh admissible for this battery. This rescales the value of the reward
    """
    assert delta_soh >= 0, "Delta_soh should be non negative"
    r = 0
    r += compute_buy_sell_reward(buy_price=buy_price, sell_price=sell_price, grid_energy=grid_energy)
    r -= delta_soh * battery_cost / (1 - soh_limit)
    assert delta_soh * battery_cost / (1 - soh_limit) >= 0,  "The delta battery cost should be non-negative"
    return r


def linearized_deg_reward(buy_price, sell_price, battery_cost, grid_energy, delta_deg, deg_limit):
    assert delta_deg >= 0, "delta_deg be non-negative"
    assert deg_limit >= 0, "deg_limit should be non-negative"
    assert battery_cost >= 0, "battery_cost should be non-negative"
    assert buy_price >= 0, "buy_price should be non-negative"
    assert sell_price >= 0, "sell_price should be non-negative"
    r = 0
    r += compute_buy_sell_reward(buy_price=buy_price, sell_price=sell_price, grid_energy=grid_energy)
    r -= delta_deg * battery_cost / deg_limit
    assert delta_deg * battery_cost / deg_limit >= 0, "The delta battery cost should be non-negative"
    return r


def compute_buy_sell_reward(buy_price, sell_price, grid_energy):
    r = 0
    if grid_energy < 0:
        r += sell_price * abs(grid_energy)
        assert sell_price * abs(grid_energy) >= 0, "value is {}".format(sell_price * abs(grid_energy))
    else:
        r -= buy_price * abs(grid_energy)
        assert buy_price * abs(grid_energy) >= 0, "value is {}".format(buy_price * abs(grid_energy))
    return r
