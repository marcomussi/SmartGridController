import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def open_results(base_path):
    fixed_action_results = None
    with open(os.path.join(base_path, "result.json")) as f1:
        agent_results = json.load(f1)
    with open(os.path.join(base_path, "result_kpi.json")) as f2:
        baselines_results = json.load(f2)
    try:
        with open(os.path.join(base_path, "result_fixed_action.json")) as f3:
            fixed_action_results = json.load(f3)
    except:
        print("No fixed action file found")
    return agent_results, baselines_results, fixed_action_results


def open_postfix_results(base_path, postfixes, names):
    results = list()
    new_names = list()
    for i in range(len(postfixes)):
        postfix = postfixes[i]
        name = names[i]
        path = os.path.join(base_path, "result{}.json".format(postfix))
        try:
            with open(path) as f:
                results.append(json.load(f))
                new_names.append(name)
        except:
            print("Not able to open {}".format(path))
    return results, new_names


def get_mean_history(histories):
    min_lenght = len(histories[0])
    for history in histories:
        if len(history) < min_lenght:
            min_lenght = len(history)
    truncated_hist = [x[0:min_lenght] for x in histories]
    array_histories = np.array([np.array(x) for x in truncated_hist])
    mean_history = np.mean(array_histories, axis=0)
    return mean_history


def get_mean_history_dict(agent_results, baselines_results, fixed_action_results, key):
    agent_histories = agent_results["AGENT"][key]["history"]
    agent_mean_history = get_mean_history(agent_histories)
    mean_history_dict = dict()
    mean_history_dict["AGENT"] = agent_mean_history
    for base_name in baselines_results:
        base_histories = baselines_results[base_name][key]["history"]
        base_mean_history = get_mean_history(base_histories)

        mean_history_dict[base_name] = base_mean_history
    return mean_history_dict


def get_frequency(agent_json):
    """The metric_json contains the frequency expressed as samples wrt a t_sample"""
    return agent_json["AGENT"]["J"]["frequency"]


def get_frequency_samples(agent_json):
    episode_lengths = [len(x) for x in agent_json["AGENT"]["J"]["history"]]
    return min(episode_lengths)


def get_n_episodes(agent_json):
    return len(agent_json["AGENT"]["J"]["values"])


def print_best_strategy(base_path, key):
    if key not in ["J", "reward", "battery_cost", "energy_profit", "delta_soh"]:
        raise Exception("No metric called {}".format(key))
    print("Printing each value of {} and find best".format(key))
    agent_results, baselines_results, fixed_action_results = open_results(base_path)
    # agent mean
    agent_mean_value = agent_results["AGENT"][key]["mean"]
    print("{} -- mean_{} = {}".format("AGENT", key, agent_mean_value))
    means = dict()
    means["AGENT"] = agent_mean_value
    # filter baselines
    for baseline_name in baselines_results:
        baseline_mean = baselines_results[baseline_name][key]["mean"]
        print("{} -- mean_{} = {}".format(baseline_name, key, baseline_mean))
        means[baseline_name] = baseline_mean
    # find maximum
    best = None
    for name, item in means.items():
        if best is None:
            best = name
        else:
            if key != "delta_soh":
                if means[name] > means[best]:
                    best = name
            elif key == "delta_soh":
                if means[name] < means[best]:
                    best = name
            else:
                raise Exception("weird behaviour")

    print("The best strategy was {}".format(best))


def visualize(base_path, key):
    if key not in ["J", "reward", "battery_cost", "energy_profit", "delta_soh"]:
        raise Exception("No metric called {}".format(key))
    agent_results, baselines_results, fixed_action_results = open_results(base_path)
    # agent history
    mean_history_dict = get_mean_history_dict(agent_results, baselines_results, fixed_action_results, key)
    # visualization
    fig, ax = plt.subplots()
    legend_names = list()
    frequency = agent_results["AGENT"][key]["frequency"]
    agent_mean_history = mean_history_dict["AGENT"]
    time = np.arange(0, len(agent_mean_history)) * frequency / 24
    for baseline_name, mean_history in mean_history_dict.items():
        legend_names.append(baseline_name)
        ax.plot(time, mean_history)
    ax.legend(legend_names)
    ax.set_xlabel("Time (days)")
    if key == "J" or key == "reward":
        ax.set_title("Cumulative reward (€)")
    elif key == "battery_cost":
        ax.set_title("Battery cost (€)")
    elif key == "energy_profit":
        ax.set_title("Energy profit (€)")
    elif key == "delta_soh":
        ax.set_title("Delta SOH (1)")
    else:
        raise Exception("wierd behaviour with title")
    plt.show()


def visualize_differential(base_path, key, reference=None):
    kpis_list = ["J", "reward", "battery_cost", "energy_profit", "delta_soh"]
    if reference is not None:
        assert reference in ["Soc20-80", "Only_grid", "Only_battery"], "reference kpi not valid"
    if key not in kpis_list:
        raise Exception("No metric called {}".format(key))
    agent_results, baselines_results, fixed_action_results = open_results(base_path)
    mean_history_dict = get_mean_history_dict(agent_results, baselines_results, fixed_action_results, key)
    print(mean_history_dict.keys())
    history_names = list()
    history_list = list()
    for history_name, value in mean_history_dict.items():
        history_names.append(history_name)
        history_list.append(value)
    history_arr = np.array(history_list)
    if key == "delta_soh":
        base_history_arr = np.max(history_arr, axis=0)
    else:
        base_history_arr = np.min(history_arr, axis=0)
    for i in range(len(history_list)):
        history_list[i] = np.abs(np.array(history_list[i]) - base_history_arr)

    for history in history_list:
        plt.plot(history)
    plt.title("{} - differential plot".format(key))
    plt.legend(history_names)
    plt.show()


def visualize_same_figure(base_path, t_sample=3600, diff=False, mult_soh=None, save=False, filename="same_fig",
                          use_baseline=False):
    agent_results, baselines_results, fixed_action_results = open_results(base_path)
    keys = ["reward", "battery_cost", "energy_profit", "delta_soh"]
    readable_keys = {
        "reward": "Profit",
        "battery_cost": "Battery Cost",
        "energy_profit": "Energy Profit",
        "delta_soh": "Degradation"
    }
    readable_names = {
        "AGENT": "Agent",
        "Soc20-80": "SoC20-80",
        "Only_battery": "OnlyBattery",
        "Only_grid": "OnlyGrid"
    }
    agent_results, baselines_results, fixed_action_results = open_results(base_path)
    # frequency is expressed in hours
    frequency = get_frequency(agent_results)
    n_freq_samples = get_frequency_samples(agent_results)
    days = np.arange(0, n_freq_samples) * frequency / 24
    fig, axs = plt.subplots(2, 2)

    for plot_index in range(len(keys)):
        key = keys[plot_index]
        mean_history_dict = get_mean_history_dict(agent_results, baselines_results, fixed_action_results, key)
        history_names = list()
        history_list = list()
        for history_name, value in mean_history_dict.items():
            history_names.append(history_name)
            history_list.append(value)
            if use_baseline and history_name == "Soc20-80":
                base_history_arr = value
        history_arr = np.array(history_list)
        # differential visualization
        if diff and not use_baseline:
            if key == "delta_soh":
                base_history_arr = np.max(history_arr, axis=0)
            else:
                base_history_arr = np.min(history_arr, axis=0)
        if diff or use_baseline:
            for i in range(len(history_list)):
                if key != "delta_soh":
                    history_list[i] = np.array(history_list[i]) - base_history_arr
                else:
                    history_list[i] = base_history_arr - np.array(history_list[i])

        ax = axs.flat[plot_index]
        for history in history_list:
            if key == "delta_soh":
                if mult_soh is not None:
                    history = history * mult_soh
            ax.plot(days, history)
        ax.set_xlabel("Time (days)")
        if key == "delta_soh":
            unit_measure = "1"
        else:
            unit_measure = "€"

        ax.set_ylabel("{} ({})".format(readable_keys[key], unit_measure))

    legend_names = [readable_names[name] for name in history_names]
    fig.legend(legend_names)
    fig.suptitle("Mean KPIs over {} days, {} episodes".format(int(days[-1]), get_n_episodes(agent_results)))
    fig.tight_layout()
    if save:
        fig.savefig(filename)
    plt.show()
    return fig


def hist_action_profile(base_path, profile_number, postfix=None, save=False):
    # real actions
    if postfix is None:
        with open(os.path.join(base_path, "actions.pkl"), "rb") as f:
            agent_action_profiles = pickle.load(f)
    else:
        with open(os.path.join(base_path, "actions{}.pkl".format(postfix)), "rb") as f:
            agent_action_profiles = pickle.load(f)

    action_profile = agent_action_profiles[profile_number]
    edge_bin2 = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    fig, (ax1, ax2) = plt.subplots(2)
    a, _, _ = ax1.hist(action_profile, bins=edge_bin2, align="mid", edgecolor="white")
    ax1.set_title("Actions Distribution")
    ax1.set_xlabel("Actions")
    ax1.set_ylabel("Occurrences")
    # actuated actions
    with open(os.path.join(base_path, "actuated_actions.pkl"), "rb") as f:
        actuated_action_profiles = pickle.load(f)

    actuated_profile = actuated_action_profiles[profile_number] * 10
    edge_bin2 = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    c, _, _ = ax2.hist(actuated_profile, bins=edge_bin2, align="mid", edgecolor="white", color="tab:orange")
    ax2.set_title("Actuated Actions Distribution")
    ax2.set_xlabel("Actions")
    ax2.set_ylabel("Occurrences")
    for ax in (ax1, ax2):
        ax.label_outer()
    max_y = np.max([a, c])
    eps = 1000
    ax1.set_ylim([0, max_y + eps])
    ax2.set_ylim([0, max_y + eps])
    fig.tight_layout()
    # fine grained
    max_value = 10
    n_samples = 51
    # edge_bin_fine_grained = np.linspace(0, max_value, n_samples) - max_value / (n_samples - 1) / 2
    # plt.figure()
    # _ = plt.hist(actuated_profile, bins=edge_bin_fine_grained, align="mid", edgecolor="white")
    # plt.title("Fine grained actuated")
    # difference in actuation
    df = pd.DataFrame()
    df["agent"] = a
    df["actuated"] = c
    df.plot.bar(rot=0)
    if save:
        plt.savefig("comparison_hist.png")
    plt.show()


def plot_action_profiles(base_path, profile_number, start_index, period):
    # real actions
    with open(os.path.join(base_path, "actions.pkl"), "rb") as f:
        agent_action_profiles = pickle.load(f)
    # actuated actions
    with open(os.path.join(base_path, "actuated_actions.pkl"), "rb") as f:
        actuated_action_profiles = pickle.load(f)

    fig, ax1 = plt.subplots()
    agent_profile = agent_action_profiles[profile_number][start_index:start_index + period] / 10
    actuated_profile = actuated_action_profiles[profile_number][start_index:start_index + period]
    print(len(agent_profile))

    color = 'tab:blue'
    ax1.set_xlabel('time')
    ax1.set_ylabel("Desired action", color=color)
    ax1.plot(agent_profile, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel("Actuated action", color=color)  # we already handled the x-label with ax1
    ax2.plot(actuated_profile, color=color)
    # ax2.plot(np.zeros(len(environment.net_profile[1:len(soc_profile)])), color="green")
    ax2.tick_params(axis='y', labelcolor=color)

    """ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax3.set_ylabel("action", color=color)  # we already handled the x-label with ax1
    ax3.plot(action_profile, color=color)
    ax3.tick_params(axis='y', labelcolor=color)"""

    fig.tight_layout()  # without this, the right y-label is slightly clipped
    plt.show()


def plot_fit_agents(base_path, key, postfixes, names, diff=False):
    if key not in ["J", "reward", "battery_cost", "energy_profit", "delta_soh"]:
        raise Exception("No metric called {}".format(key))
    # agent history
    results, names = open_postfix_results(base_path, postfixes, names)
    mean_history_dict = get_mean_history_multiple_fit(results, names, key)
    history_names, history_list = get_history_names_arr(mean_history_dict, key, diff=diff)
    time_arr = create_time_array(results[0])
    fig, ax = plt.subplots()
    for history in history_list:
        ax.plot(time_arr, history)
    fig.legend(history_names)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(filter_y_label(key))
    ax.set_title("{} of agent at different fit steps".format(filter_key(key)))
    plt.show()


def plot_fit_agent_same_figure(base_path, postfixes, names, diff=False):
    keys = ["J", "battery_cost", "energy_profit", "delta_soh"]
    results, names = open_postfix_results(base_path, postfixes, names)
    fig, axs = plt.subplots(2, 2)
    time_arr = create_time_array(results[0])
    for index in range(len(keys)):
        key = keys[index]
        ax = axs.flat[index]
        mean_history_dict = get_mean_history_multiple_fit(results, names, key)
        history_names, history_list = get_history_names_arr(mean_history_dict, key, diff=diff)
        for history in history_list:
            ax.plot(time_arr, history)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(filter_y_label(key))
        ax.set_title("{}".format(filter_key(key)))
    fig.suptitle("KPIs at fit steps")
    fig.tight_layout()
    fig.legend(names)
    plt.show()


def get_mean_history_multiple_fit(results, names, key):
    mean_history_dict = dict()
    for index in range(len(results)):
        result = results[index]
        name = names[index]
        history = result["AGENT"][key]["history"]
        mean_history = get_mean_history(history)
        mean_history_dict[name] = mean_history

    return mean_history_dict


def get_history_names_arr(mean_history_dict, key, diff=False):
    history_names = list()
    history_list = list()
    for history_name, value in mean_history_dict.items():
        history_names.append(history_name)
        history_list.append(value)
    history_arr = np.array(history_list)
    # differential visualization
    if diff:
        if key == "delta_soh":
            base_history_arr = np.max(history_arr, axis=0)
        else:
            base_history_arr = np.min(history_arr, axis=0)
        for i in range(len(history_list)):
            history_list[i] = np.abs(np.array(history_list[i]) - base_history_arr)
    return history_names, history_list


def create_time_array(result):
    frequency = get_frequency(result)
    n_freq_samples = get_frequency_samples(result)
    days = np.arange(0, n_freq_samples) * frequency / 24
    return days


def filter_y_label(key):
    if key == "J" or key == "reward":
        label = "Cumulative reward (€)"
    elif key == "battery_cost":
        label = "Battery cost (€)"
    elif key == "energy_profit":
        label = "Energy profit (€)"
    elif key == "delta_soh":
        label = "Delta SOH (1)"
    else:
        raise Exception("key not supported")
    return label


def filter_key(key):
    if key == "J" or key == "reward":
        label = "Cumulative reward"
    elif key == "battery_cost":
        label = "Battery cost"
    elif key == "energy_profit":
        label = "Energy profit"
    elif key == "delta_soh":
        label = "Delta SOH"
    else:
        raise Exception("key not supported")
    return label
