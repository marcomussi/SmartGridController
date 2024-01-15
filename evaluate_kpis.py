import argparse
import json
import os
import pickle
import time

import numpy as np
from mushroom_rl.core import Logger

from rse_lib.utils.CoreTester import CoreTester


def load_json(res_folder):
    # load the configuration file
    files = os.listdir(res_folder)
    json_name = [x for x in files if x.endswith(".json") and x != "result.json" and x != "result_kpi.json" and
                 x != "result_fixed_action.json" and "result" not in x][0]
    json_path = os.path.join(res_folder, json_name)
    with open(json_path, "r") as file:
        exp_params = json.load(file)
    return exp_params


def get_agent_path(res_folder):
    files = os.listdir(res_folder)
    agent_file = [x for x in files if x.endswith(".msh")][0]
    ag_path = os.path.join(res_folder, agent_file)
    return ag_path


if __name__ == "__main__":
    # the parser expects the presence of a configuration file for the environment
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--exp-folder", type=str, help="folder where configuration and agent are stored")
    parser.add_argument("--jobs", type=int, help="number of jobs used to build the extra trees")
    parser.add_argument("--freq", type=int, help="This argument fixies the evaluation every freq days")
    parser.add_argument("--years", type=int, help="This argument changes the evaluation year and overrides "
                                                  "the one in the configuration file")
    parser.add_argument("--fixed-action", default=False, action="store_true")
    parser.add_argument("--n-actions", type=int, help="Number of fixed actions to set", default=11)
    parser.add_argument("--env-print", default=False, action="store_true")
    args = parser.parse_args()
    result_folder = args.exp_folder
    n_jobs = args.jobs
    freq = args.freq
    n_year_arg = args.years
    is_fixed_action = args.fixed_action
    n_actions = args.n_actions
    print_env = args.env_print

    if result_folder is None:
        parser.error("--exp-folder should not be None")
    if n_jobs <= 0 or n_jobs is None:
        parser.error("--jobs argument should be a positive integer. It was {} instead".format(n_jobs))

    # load the configuration file
    experiment_params = load_json(result_folder)
    experiment_params["environment"]["action_type"] = "continuous"
    # parameters
    experiment_name = experiment_params["name"]
    SEED = experiment_params["test_seed"]
    test_params = experiment_params["test_params"]
    # create logging for the current experiment
    logger = Logger("evaluation", results_dir=result_folder, log_console=True,
                    seed=SEED)
    if print_env:
        logger.info("Parameters of the experiment\n{}".format(json.dumps(experiment_params, indent=2)))
    # overriding n_years
    if n_year_arg is not None:
        logger.warning("Changing the n_years field from {} to {}. This is because the --years option was given through"
                       " command line".format(experiment_params["environment"]["n_years"], n_year_arg))
        experiment_params["environment"]["n_years"] = n_year_arg
    # check if there is already a result json
    if is_fixed_action:
        save_metrics_path = os.path.join(result_folder, "result_fixed_action.json")
    else:
        save_metrics_path = os.path.join(result_folder, "result_kpi.json")

    if os.path.exists(save_metrics_path):
        logger.warning("CAREFULL! There's already a result for this agent, it will be overwritten!")

    env_params = experiment_params["environment"]
    # number of parallel simulation to run for each baseline
    n_episodes = test_params["n_episodes"]
    # creation of the tester for evaluation
    buy_price = env_params["buy_price"]
    sell_price = env_params["sell_price"]
    batt_cost = env_params["battery_capacity"] * env_params["nominal_voltage"] * env_params["unit_battery_cost"]
    soh_limit = env_params["min_soh"]
    tester = CoreTester(experiment_name, test_params["is_episode"], SEED,
                        n_episodes=test_params["n_episodes"], n_steps=test_params["n_steps"])
    tester.load_default_kpis(buy_price=buy_price, sell_price=sell_price, batt_cost=batt_cost, soh_limit=soh_limit)
    frequency = None
    if freq is not None:
        frequency = freq * 24 * 3600 // env_params["t_sample"]
        logger.info("KPI values frequency calculation every {} samples".format(frequency))

    # creation of n_episodes environments and testing
    logger.info("Testing using {} jobs".format(n_jobs))
    baselines_dict = dict()
    action_dict = dict()
    reward_dict = dict()
    actuated_dict = dict()
    env_trans_dict = dict()
    additional_metrics = dict()
    start = time.time()
    experiment_params["environment"]["action_type"] = "continuous"
    policy_types = ["Soc20-80", "Only_battery", "Only_grid"]
    if is_fixed_action:
        experiment_params["environment"]["action_type"] = "continuous"
        action_values = np.arange(0, n_actions) / (n_actions - 1)
        logger.info("Fixed action values = {}".format(action_values))
        logger.info("Parallel creation AND testing")
        logger.info("Started parallel testing")
        for action_value in action_values:
            logger.info("Action value: {}".format(action_value))
            tester_dict, list_of_transitions, additional_data = \
                tester.parallel_creation_and_testing(n_tests=tester.n_episodes,
                                                     experiment_params=experiment_params,
                                                     alg="fixed_action",
                                                     agent_path=None,
                                                     n_jobs=n_jobs,
                                                     frequency=frequency,
                                                     baseline_type=None,
                                                     action=action_value)
            baselines_dict[action_value] = tester_dict
            action_dict[action_value] = additional_data["actions_list"]
            reward_dict[action_value] = additional_data["rewards_list"]
            actuated_dict[action_value] = additional_data["actuated_list"]
            env_trans_dict[action_value] = list_of_transitions
            additional_metrics[action_value] = additional_data
    else:
        logger.info("Parallel creation AND testing")
        logger.info("Started parallel testing")
        for policy_type in policy_types:
            logger.info("Testing baseline {}".format(policy_type))
            n_tests = tester.n_episodes if tester.is_episode else 1
            tester_dict, list_of_transitions, additional_data = \
                tester.parallel_creation_and_testing(n_tests=n_tests,
                                                     experiment_params=experiment_params,
                                                     alg="baselines",
                                                     agent_path=None,
                                                     n_jobs=n_jobs,
                                                     frequency=frequency,
                                                     baseline_type=policy_type)
            baselines_dict[policy_type] = tester_dict
            action_dict[policy_type] = additional_data["actions_list"]
            reward_dict[policy_type] = additional_data["rewards_list"]
            actuated_dict[policy_type] = additional_data["actuated_list"]
            env_trans_dict[policy_type] = list_of_transitions
            additional_metrics[policy_type] = additional_data
    # end of testing
    logger.info("testing done in {} seconds".format(time.time() - start))
    for baseline_name, b_dict in baselines_dict.items():
        logger.info("BASELINE {}".format(baseline_name))
        for metric_name, metric_dict in b_dict.items():
            logger.info(
                "{}: mean = {} -- min = {} -- max = {}".format(metric_name, metric_dict["mean"], metric_dict["min"],
                                                               metric_dict["max"]))
        logger.weak_line()
    # save metrics
    logger.info("Saving results")
    with open(save_metrics_path, "w") as f:
        json.dump(baselines_dict, f, indent=2)
    logger.info("results saved at path {}".format(save_metrics_path))

    try:
        for policy_type, transitions in env_trans_dict.items():
            transitions_pickle_path = os.path.join(result_folder, "transitions_{}.pkl".format(policy_type))
            with open(transitions_pickle_path, "wb") as transition_file:
                pickle.dump(transitions, transition_file)
            logger.info("Save list of transitions at {}".format(transitions_pickle_path))
    except:
        print("Some problem while saving transitions distribution")

    for policy_type, actions_list in action_dict.items():
        action_list = [x.flatten() for x in actions_list]
        action_pickle_path = os.path.join(result_folder, "actions_{}.pkl".format(policy_type))
        with open(action_pickle_path, "wb") as action_file:
            pickle.dump(action_list, action_file)
        logger.info("Saved action file at {}".format(action_pickle_path))

    logger.info("Creating reward pickle file")
    for policy_type, reward_list in reward_dict.items():
        reward_pickle_path = os.path.join(result_folder, "rewards_{}.pkl".format(policy_type))
        with open(reward_pickle_path, "wb") as reward_file:
            pickle.dump(reward_list, reward_file)
        logger.info("Saved reward file at {}".format(reward_pickle_path))

    logger.info("Creating actuated action pickle file")
    for policy_type, actuated_list in actuated_dict.items():
        actuated_pickle_path = os.path.join(result_folder, "actuated_actions_{}.pkl".format(policy_type))
        with open(actuated_pickle_path, "wb") as actuated_file:
            pickle.dump(actuated_list, actuated_file)
        logger.info("Saved actuated actions file at {}".format(actuated_pickle_path))

    logger.info("creating degradation factor pickle files")
    for policy_type, additional_metric_dict in additional_metrics.items():
        for deg_key in ["f_cal", "f_cycle", "stream_f_cycle_past"]:
            deg_list = additional_metric_dict[deg_key]
            deg_path = os.path.join(result_folder,"{}_{}.pkl".format(deg_key, policy_type))
            with open(deg_path, "wb") as deg_file:
                pickle.dump(deg_list, deg_file)

