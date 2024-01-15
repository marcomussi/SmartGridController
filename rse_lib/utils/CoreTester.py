import time

import numpy as np
from joblib import Parallel, delayed
from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core, Logger, Agent
from mushroom_rl.utils import dataset
from mushroom_rl.utils.parameters import Parameter

from rse_lib.environment.MushroomBatteryEnv import MushroomBatteryEnv
from rse_lib.utils.callbacks import LinearDegradationCallback, StepCallbackList, SoHCallback
from rse_lib.utils.kpi import J_KPI, GridBatteryKPI, EnergyCostKPI, BatteryCostKPI, DeltaSohKPI
from rse_lib.utils.mushroom_wrappers.MushSocPolicy import MushSocPolicy, MushFixedAction
from rse_lib.utils.state import BatteryStateTracker


class CoreTester:
    LOG_DIR = "./experiment_folder/logs/"

    def __init__(self, exp_name, is_episode, seed, n_episodes=None, n_steps=None, log_dir=None):
        self.is_episode = is_episode
        self.n_episodes = None
        self.n_steps = None
        self.name = exp_name
        if self.is_episode:
            assert n_episodes is not None, "episodic testing needs number of episodes"
            self.n_episodes = n_episodes
        else:
            assert n_steps is not None, "non episodic testing needs number of testing steps"
            self.n_steps = n_steps
        self.cores = {}
        self.envs = {}
        self.seed = seed
        self.results_logger = Logger(self.name + "_results", results_dir=CoreTester.LOG_DIR,
                                     log_console=True, seed=seed, append=False)
        self.kpi_dict = {}

    def register_core(self, name, core, env):
        self.cores[name] = core
        self.envs[name] = env

    def add_kpi(self, name, kpi):
        self.kpi_dict[name] = kpi

    def load_default_kpis(self, buy_price, sell_price, batt_cost, soh_limit):
        j_kpi = J_KPI()
        reward_kpi = GridBatteryKPI(buy_price=buy_price, sell_price=sell_price,
                                    batt_cost=batt_cost,
                                    soh_limit=soh_limit)
        energy_kpi = EnergyCostKPI(buy_price=buy_price, sell_price=sell_price)
        battery_kpi = BatteryCostKPI(batt_cost=batt_cost, soh_limit=soh_limit)
        delta_soh_kpi = DeltaSohKPI(soh_limit=soh_limit)
        self.add_kpi("J", j_kpi)
        self.add_kpi("reward", reward_kpi)
        self.add_kpi("battery_cost", battery_kpi)
        self.add_kpi("energy_profit", energy_kpi)
        self.add_kpi("delta_soh", delta_soh_kpi)

    def evaluate_all(self, **kwargs):
        metrics_dict = {}
        for name in self.cores:
            core_metrics = self.evaluate_core(name)
            metrics_dict[name] = core_metrics
            self.results_logger.strong_line()
            self.results_logger.info("Experiment {}".format(name))
            if self.is_episode:
                for metric in core_metrics:
                    number_of_episodes = core_metrics[metric]["n_episodes"]
                    min_val = core_metrics[metric]["min"]
                    max_val = core_metrics[metric]["max"]
                    mean_val = core_metrics[metric]["mean"]
                    self.results_logger.info(
                        "Metric: {} -- mean = {} \tmin = {} \tmax={}".format(metric, mean_val, min_val, max_val))
                    self.results_logger.weak_line()
            else:
                for metric in core_metrics:
                    self.results_logger.info("{} = {}".format(metric, core_metrics[metric]["value"]))
                    self.results_logger.weak_line()

        return metrics_dict

    def evaluate_core(self, name):
        core = self.cores[name]
        env = self.envs[name]
        env.seed(self.seed)
        gamma = env.info.gamma
        metrics = {}
        for kpi_name in self.kpi_dict:
            metrics[kpi_name] = {"values": []}
        if self.is_episode:
            for i in range(self.n_episodes):
                start_time = time.time()
                episode_dataset = core.evaluate(n_episodes=1)
                end_time = time.time()
                self.results_logger.info("Episode {} completed in {}".format(i, end_time - start_time))
                for kpi_name in self.kpi_dict:
                    kpi = self.kpi_dict[kpi_name]
                    kpi_value = kpi.get_kpi_value(episode_dataset, env)
                    metrics[kpi_name]["values"].append(kpi_value)

            for metric_name in metrics:
                values = metrics[metric_name]["values"]
                metrics[metric_name]["min"] = np.min(values)
                metrics[metric_name]["max"] = np.max(values)
                metrics[metric_name]["mean"] = np.mean(values)
                metrics[metric_name]["value"] = metrics[metric_name]["mean"]
                metrics[metric_name]["n_episodes"] = len(values)
        else:
            # raise Warning("Not implemented correct episode evaluation in the step check")
            step_dataset = core.evaluate(n_steps=self.n_steps)

            for kpi_name in self.kpi_dict:
                kpi = self.kpi_dict[kpi_name]
                kpi_value = kpi.get_kpi_value(step_dataset, env)
                metrics[kpi_name]["value"] = kpi_value
        return metrics

    def get_datasets(self, name):
        core = self.cores[name]
        env = self.envs[name]
        env.seed(self.seed)
        gamma = env.info.gamma
        dataset_list = list()
        if self.is_episode:
            for i in range(self.n_episodes):
                start_time = time.time()
                episode_dataset = core.evaluate(n_episodes=1)
                end_time = time.time()
                self.results_logger.info("Episode {} completed in {}".format(i, end_time - start_time))
                dataset_list.append(episode_dataset)
        else:
            step_dataset = core.evaluate(n_steps=self.n_steps)
            dataset_list.append(step_dataset)
        return dataset_list

    def parallel_testing(self, n_jobs, seed, frequency=None):
        """
        Test in parallel all the environments considered
        :param n_jobs: number of jobs run in parallel
        :param seed: starting seed of the environments
        :param frequency: sample frequency at which the kpi are computed
        :return: a dictionary in the form of ["kpi1":[list_of_values], "frequency":frequency,
        "kpi1_history":[ [] ] <- list of lists kpi values for each episode}
        """
        # creation of a list of environments and the respective list of cores
        env_list = list()
        core_list = list()
        keys_list = list(self.cores.keys())
        keys_list.sort()  # sorting should not be relevant, but it just gives a clearer order
        seed_index = 0
        delayed_funcs = []
        for env_core_name in keys_list:
            env = self.envs[env_core_name]
            core = self.cores[env_core_name]
            env_list.append(env)
            core_list.append(core)
            eval_seed = seed + seed_index
            seed_index += 1
            delayed_funcs.append(delayed(parallel_core_evaluation)(env=env,
                                                                   core=core,
                                                                   seed=eval_seed,
                                                                   is_episode=self.is_episode,
                                                                   n_steps=self.n_steps))
        # now parallel call on evaluation
        return_arguments = Parallel(n_jobs=n_jobs, backend="loky")(delayed_funcs)
        evaluation_results, list_of_transitions, additional_data = \
            self.compute_metrics(return_arguments, frequency=frequency)
        return evaluation_results, list_of_transitions, additional_data

    def compute_metrics(self, return_arguments, frequency=None):
        list_of_transitions = [x[0] for x in return_arguments]
        env_list = [x[1] for x in return_arguments]
        core_list = [x[2] for x in return_arguments]
        additional_metrics = [x[3] for x in return_arguments]
        soh_lists = list()
        for metric_dict in additional_metrics:
            soh_lists.extend(metric_dict["SoH"])
        # evaluation part
        evaluation_results = dict()
        for kpi_name, kpi in self.kpi_dict.items():
            kpi_values = []
            list_of_histories = []
            t_sample = None
            for episode_index in range(len(list_of_transitions)):
                # first compute kpi value on whole episode
                episode_transitions = list_of_transitions[episode_index]
                episode_soh = soh_lists[episode_index]
                env = env_list[episode_index]
                kpi_value = kpi.get_kpi_value(data_set=episode_transitions, env=env, soh_values=episode_soh)
                kpi_values.append(kpi_value)
                if frequency is not None:
                    kpi_histories = []
                    for freq_index in range(0, len(episode_transitions), frequency):
                        if freq_index == 0:
                            kpi_histories.append(0)
                        else:
                            """
                            # TODO questa linea era per risolvere un bug ma non ha avuto nessun effetto
                            if freq_index >= len(episode_transitions[0:freq_index]) \
                                    or freq_index >= episode_soh[0:freq_index]:
                                freq_index = min(len(episode_transitions[0:freq_index]), len(episode_soh[0:freq_index]))
                            """
                            kpi_histories.append(kpi.get_kpi_value(data_set=episode_transitions[0:freq_index], env=env,
                                                                   soh_values=episode_soh[0:freq_index]))
                    list_of_histories.append(kpi_histories)
            # adds an entry of the type "name of the kpi":list_of_values_for_each_episode
            kpi_dict = dict()
            kpi_dict["values"] = kpi_values
            kpi_dict["mean"] = np.mean(kpi_values)
            kpi_dict["min"] = np.min(kpi_values)
            kpi_dict["max"] = np.max(kpi_values)
            if list_of_histories is not None:
                kpi_dict["frequency"] = frequency
                kpi_dict["history"] = list_of_histories
            # other information
            kpi_dict["t_sample"] = env_list[0].env.t_sample
            kpi_dict["n_years"] = env_list[0].env.n_years
            evaluation_results[kpi_name] = kpi_dict
        # extract reward and action profiles
        actions_list = []
        rewards_list = []
        for transitions in list_of_transitions:
            _, actions, rewards, _, _, _ = dataset.parse_dataset(transitions)
            actions_list.append(actions)
            rewards_list.append(rewards)
        # return also the real actuated actions
        actuated_actions_list = []
        for environment in env_list:
            actuated_actions_list.append(environment.env.real_action_history)

        # creation of the dictionary that contains all the long profiles
        return_dict = dict()
        for key in additional_metrics[0].keys():
            return_dict[key] = list()
        # fill with additional metric
        for metric_dict in additional_metrics:
            for key in metric_dict.keys():
                return_dict[key].extend(metric_dict[key])
        # add all lists and put them in a dictionary
        return_dict["actions_list"] = actions_list
        return_dict["rewards_list"] = rewards_list
        return_dict["actuated_list"] = actuated_actions_list
        return evaluation_results, list_of_transitions, return_dict

    def parallel_creation_and_testing(self, n_tests, experiment_params, alg, agent_path, n_jobs, frequency,
                                      baseline_type=None, action=None):
        delayed_funcs = []
        for test_index in range(n_tests):
            # shallow copy of dictionary
            exp_par_copy = experiment_params.copy()
            env_seed = self.seed + test_index
            algorithm = alg
            path = agent_path
            is_episode = self.is_episode
            n_steps = self.n_steps
            delayed_funcs.append(delayed(parallel_creation_and_eval)(experiment_params=exp_par_copy,
                                                                     seed=env_seed,
                                                                     alg=algorithm,
                                                                     agent_path=path,
                                                                     is_episode=is_episode,
                                                                     n_steps=n_steps,
                                                                     baseline_type=baseline_type,
                                                                     action=action))
        # now parallel call on evaluation
        return_arguments = Parallel(n_jobs=n_jobs, backend="loky")(delayed_funcs)
        evaluation_results, list_of_transitions, additional_data = self.compute_metrics(
            return_arguments=return_arguments,
            frequency=frequency)
        return_dict = dict()
        return evaluation_results, list_of_transitions, additional_data


def parallel_core_evaluation(env, core, seed, is_episode, n_steps=None):
    env.seed(seed)
    # if the core evaluates on episodes, do just one episode, otherwise test for n_steps
    if is_episode:
        transitions = core.evaluate(n_episodes=1)
    else:
        transitions = core.evaluate(n_steps=n_steps)
    return transitions, env, core


def parallel_creation_and_eval(experiment_params, seed, alg, agent_path, is_episode, n_steps, xgboost_jobs=5,
                               baseline_type=None, action=None):
    name = experiment_params["name"]
    environment_params = experiment_params["environment"]
    use_logger = experiment_params["use_logger"]
    env = MushroomBatteryEnv(name=name, seed=seed, experiment_params=environment_params, use_logger=use_logger)
    env.seed(seed)
    if alg == "fqi":
        agent = FQI.load(agent_path)
        if experiment_params["approximator_type"] in ["xgboost", "xcboost"]:
            for regr in agent.approximator._impl.model:
                regr._Booster.set_param('n_jobs', xgboost_jobs)
        deterministic_epsilon = Parameter(0.)
        agent.policy.set_epsilon(deterministic_epsilon)
    elif alg == "baselines":
        if baseline_type == "Soc20-80":
            indexes = env.env.state_tracker.get_indexes()
            soc_index = indexes[BatteryStateTracker.SOC]
            c_rate_idx = None
            net_pv_idx = None
            try:
                c_rate_idx = indexes[BatteryStateTracker.C_RATE]
            except KeyError:
                try:
                    net_pv_idx = indexes[BatteryStateTracker.NET_POWER]
                except KeyError:
                    raise KeyError("The state does not contain c_rate nor net power, therefore it's not"
                                   "possible to find the current direction")
            charge_idx = c_rate_idx if c_rate_idx is not None else net_pv_idx
            soc_policy = MushSocPolicy(0.2, 0.8, soc_idx=soc_index, charge_var_idx=charge_idx)
            agent = Agent(env.info, soc_policy)
        elif baseline_type == "Only_grid":
            only_grid = MushFixedAction(0)
            agent = Agent(env.info, only_grid)
        elif baseline_type == "Only_battery":
            only_battery = MushFixedAction(1)
            agent = Agent(env.info, only_battery)
        else:
            raise NotImplementedError("The baselines {} is not supported.".format(baseline_type))
    elif alg == "fixed_action":
        fixed_policy = MushFixedAction(action)
        agent = Agent(env.info, fixed_policy)
    else:
        raise NotImplementedError("The alg {} is not implemented".format(alg))
    deg_callback = LinearDegradationCallback(env)
    soh_callback = SoHCallback(env, env.get_gym_env().init_soh)
    step_callback = StepCallbackList()
    # register callbacks
    step_callback.register_callback(deg_callback)
    step_callback.register_callback(soh_callback)
    core = Core(agent=agent, mdp=env, callback_step=step_callback)
    if is_episode:
        transitions = core.evaluate(n_episodes=1)
    else:
        transitions = core.evaluate(n_steps=n_steps)
    return_dict = dict()
    return_dict["f_cal"] = deg_callback.f_cal_list
    return_dict["f_cycle"] = deg_callback.f_cyc_list
    return_dict["stream_f_cycle_past"] = deg_callback.stream_f_cyc_past_list
    return_dict["SoH"] = soh_callback.soh_list
    return transitions, env, core, return_dict
