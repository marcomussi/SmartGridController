import argparse
import json
import os
import pickle
import time

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core
from mushroom_rl.core import Logger
from mushroom_rl.utils.parameters import Parameter

from rse_lib.environment.MushroomBatteryEnv import MushroomBatteryEnv
from rse_lib.utils.CoreTester import CoreTester


def load_json(res_folder):
    # load the configuration file
    files = os.listdir(res_folder)
    json_name = [x for x in files if x.endswith(".json") and "result" not in x][0]
    json_path = os.path.join(res_folder, json_name)
    with open(json_path, "r") as file:
        exp_params = json.load(file)
    return exp_params


def get_agent_path(res_folder):
    files = os.listdir(res_folder)
    agent_file = [x for x in files if x.endswith("agent.msh")][0]
    ag_path = os.path.join(res_folder, agent_file)
    return ag_path


if __name__ == "__main__":
    # the parser expects the presence of a configuration file for the environment
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--exp-folder", type=str, help="folder where configuration and agent are stored")
    parser.add_argument("--agent-path", type=str, help="explicit path of the agent that has to be tested")
    parser.add_argument("--jobs", type=int, help="number of jobs used to do the evaluation")
    parser.add_argument("--alg", type=str, help="algorithm used", choices=["fqi"])
    parser.add_argument("--freq", type=int, help="This argument fixies the evaluation every freq days")
    parser.add_argument("--years", type=int, help="This argument changes the evaluation year and overrides "
                                                  "the one in the configuration file")
    parser.add_argument("--only-test", default=False, action='store_true')
    parser.add_argument("--postfix", type=str, default="", help="Postfix to add to results files")
    parser.add_argument("--default-bolun", default=False, action="store_true", help="Ignore configuration file "
                                                                                    "degradation parameters. This "
                                                                                    "allows comparison between "
                                                                                    "different trained agents")
    parser.add_argument("--env-print", default=False, action="store_true")
    args = parser.parse_args()
    result_folder = args.exp_folder
    agent_path_arg = args.agent_path
    n_jobs = args.jobs
    alg = args.alg
    freq = args.freq
    n_year_arg = args.years
    only_test = args.only_test
    postfix = args.postfix
    default_bolun = args.default_bolun
    env_print = args.env_print

    if result_folder is None:
        parser.error("--exp-folder should not be None")
    if n_jobs <= 0 or n_jobs is None:
        parser.error("--jobs argument should be a positive integer. It was {} instead".format(n_jobs))

    # load the configuration file
    experiment_params = load_json(result_folder)

    # load agent path
    if agent_path_arg is None:
        agent_path = get_agent_path(result_folder)
    else:
        agent_path = agent_path_arg
    # parameters
    experiment_name = experiment_params["name"]
    SEED = experiment_params["test_seed"]
    test_params = experiment_params["test_params"]
    # create logging for the current experiment
    logger = Logger("evaluation", results_dir=result_folder, log_console=True,
                    seed=SEED)
    # remove from the configuration the degradation parameters
    if default_bolun:
        experiment_params["environment"]["deg_params"] = None
        logger.warning("Removed the degradation parameters from configuration dict. Default degradation parameters "
                       "are now used.")

    if env_print:
        logger.info("Parameters of the experiment\n{}".format(json.dumps(experiment_params, indent=2)))
    # overriding n_years
    if n_year_arg is not None:
        logger.warning("Changing the n_years field from {} to {}. This is because the --years option was given through"
                       " command line".format(experiment_params["environment"]["n_years"], n_year_arg))
        experiment_params["environment"]["n_years"] = n_year_arg

    # check if there is already a result json
    save_metrics_path = os.path.join(result_folder, "result{}.json".format(postfix))
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
    start = time.time()
    if only_test:
        logger.info("Only testing, no creation.")
        for episode_index in range(n_episodes):
            mdp = MushroomBatteryEnv(name=experiment_params["name"], seed=SEED + episode_index,
                                     experiment_params=experiment_params["environment"],
                                     use_logger=experiment_params["use_logger"])
            if alg == "fqi":
                agent = FQI.load(agent_path)
                agent_name = "FQI_agent"
                epsilon = Parameter(0.)
                agent.policy.set_epsilon(epsilon)
            else:
                logger.error("Cannot handle {} algorithm".format(alg))
                raise NotImplemented("need to handle different algorithm")
            core = Core(agent, mdp)
            tester.register_core(agent_name + str(episode_index), core=core, env=mdp)
        logger.info("Started parallel testing")
        tester_dict, actions_list, rewards_list, actuated_actions_list = tester.parallel_testing(n_jobs=n_jobs,
                                                                                                 seed=SEED,
                                                                                                 frequency=frequency)
    else:
        logger.info("Parallel creation AND testing")
        logger.info("Started parallel testing")
        n_tests = tester.n_episodes if tester.is_episode else 1
        tester_dict, list_of_transitions, additional_data = \
            tester.parallel_creation_and_testing(n_tests=n_tests,
                                                 experiment_params=experiment_params,
                                                 alg=alg,
                                                 agent_path=agent_path,
                                                 n_jobs=n_jobs,
                                                 frequency=frequency)
        actions_list = additional_data["actions_list"]
        rewards_list = additional_data["rewards_list"]
        actuated_actions_list = additional_data["actuated_list"]
    logger.info("testing done in {} seconds".format(time.time() - start))

    for metric_name, metric_dict in tester_dict.items():
        logger.info("{}: mean = {} -- min = {} -- max = {}".format(metric_name, metric_dict["mean"], metric_dict["min"],
                                                                   metric_dict["max"]))
    result_dict = {"AGENT": tester_dict}
    # save metrics
    logger.info("Saving results")
    with open(save_metrics_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info("results saved")
    # now creation of the files for action and rewards and dataset for future analysis
    logger.info("Creating list of transaction pickle file")
    transitions_pickle_path = os.path.join(result_folder, "transitions{}.pkl".format(postfix))
    with open(transitions_pickle_path, "wb") as transition_file:
        pickle.dump(list_of_transitions, transition_file)
    logger.info("Save list of transitions at {}".format(transitions_pickle_path))

    logger.info("Creating action pickle file")
    action_list = [x.flatten() for x in actions_list]
    action_pickle_path = os.path.join(result_folder, "actions{}.pkl".format(postfix))
    with open(action_pickle_path, "wb") as action_file:
        pickle.dump(action_list, action_file)
    logger.info("Saved action file at {}".format(action_pickle_path))
    # now rewards
    logger.info("Creating reward pickle file")
    reward_pickle_path = os.path.join(result_folder, "rewards{}.pkl".format(postfix))
    with open(reward_pickle_path, "wb") as reward_file:
        pickle.dump(rewards_list, reward_file)
    logger.info("Saved reward file at {}".format(reward_pickle_path))
    logger.info("creating actuated action file")
    actuated_path = os.path.join(result_folder, "actuated_actions{}.pkl".format(postfix))
    with open(actuated_path, "wb") as actuated_file:
        pickle.dump(actuated_actions_list, actuated_file)
    logger.info("Saved actuated actions at {}".format(actuated_path))

    # save in pickle format degradation factors
    logger.info("Saving degradation info")
    f_cal_path = os.path.join(result_folder, "f_cal{}.pkl".format(postfix))
    f_cal_list = additional_data["f_cal"]
    with open(f_cal_path, "wb") as f_cal_file:
        pickle.dump(f_cal_list, f_cal_file)
    logger.info("Saved calendarial ageing at {}".format(f_cal_path))
    # f cycle
    f_cycle_path = os.path.join(result_folder, "f_cycle{}.pkl".format(postfix))
    f_cycle_list = additional_data["f_cycle"]
    with open(f_cycle_path, "wb") as f_cycle_file:
        pickle.dump(f_cycle_list, f_cycle_file)
    logger.info("Saved cycling ageing at {}".format(f_cal_path))
    # f stream past
    f_stream_path = os.path.join(result_folder, "f_stream{}.pkl".format(postfix))
    f_stream_list = additional_data["stream_f_cycle_past"]
    with open(f_stream_path, "wb") as f_stream_file:
        pickle.dump(f_stream_list, f_stream_file)
    logger.info("Saved past cycle degradation factor at {}".format(f_stream_path))



