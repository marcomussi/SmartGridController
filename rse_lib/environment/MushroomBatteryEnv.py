import json
import math
import pickle
from abc import ABC

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.core import Logger
from pandas import DataFrame

from rse_lib.environment.BatteryEnv import BatteryEnv, init_env_from_dict


class MushroomBatteryEnv(Environment, ABC):
    """
    Wrapper of the Gym Environment used for Mushroom
    """
    LOG_DIR = "../../experiment_folder/logs/"
    SAVE_DEBUG_DIR = "./experiment_folder/results/"

    def __init__(self, name, seed, experiment_params=None, use_logger=False, state_variables=None):

        if experiment_params is None:
            self.env = BatteryEnv(seed=seed, state_variables=state_variables)
        else:
            self.env = init_env_from_dict(experiment_params, seed)
        self.seed(seed)
        obs_space = self.env.get_observation_space()
        action_space = self.env.get_action_space()
        gamma = self.env.get_gamma()
        th = self.env.get_time_horizon()
        mdp_info = MDPInfo(obs_space, action_space, gamma, th)

        self.use_logger = use_logger
        self.name = name

        if use_logger:
            self.state_logger = Logger(self.name + "_state", results_dir=MushroomBatteryEnv.LOG_DIR,
                                       log_console=True, seed=seed, append=False)
            self.action_logger = Logger(self.name + "_action", results_dir=MushroomBatteryEnv.LOG_DIR,
                                        log_console=True, seed=seed, append=False)
            self.nan_values = Logger(self.name + "_nan", results_dir=MushroomBatteryEnv.LOG_DIR,
                                     log_console=True, seed=seed, append=False)

        super().__init__(mdp_info)

    def step(self, action):
        next_state, reward, end_condition, meta_data = self.env.step(action)
        if self.use_logger:
            alpha_max = meta_data["alpha_max"] if meta_data["alpha_max"] < 1 else 1
            self.action_logger.debug(str(action) + " " + str(alpha_max))
            self.state_logger.debug(next_state)
            # find a nan in the state
            if math.isnan(next_state[5]):
                self.nan_values.debug("At iteration {} a nan c_rate was found.".format(self.env.iteration))
                # save environment in order to keep track of every variable.
                dict_env = self.env.__dict__
                try:
                    with open(MushroomBatteryEnv.SAVE_DEBUG_DIR + "environment.json", "w") as f:
                        json.dump(dict_env, f, indet=4)
                except Exception as e:
                    print("Problemi nel salvare il json")
                    print(e)

                try:
                    with open(MushroomBatteryEnv.SAVE_DEBUG_DIR + "environment.pkl", "wb") as g:
                        pickle.dump(self.env, g)
                except Exception as e:
                    print("Problemi nel salvare l'ambiente in pickle")
                    print(e)

                try:
                    net_profile = self.env.net_profile
                    df = DataFrame()
                    df["net_profile"] = net_profile
                    df.to_csv(MushroomBatteryEnv.SAVE_DEBUG_DIR + "net_profile.csv")
                except Exception as e:
                    print("Problemi a salvare il profilo netto")
                    print(e)

                try:
                    temp_profile = self.env.temp_history
                    soc_profile = self.env.soc_history
                    df = DataFrame()
                    df["temp"] = temp_profile
                    df["soc"] = soc_profile
                    df.to_csv(MushroomBatteryEnv.SAVE_DEBUG_DIR + "soc_and_temp.csv")
                except Exception as e:
                    print("Problemi a salvare soc e temp history")
                    print(e)

                try:
                    battery_dict = self.env.battery_model.__dict__
                    with open(MushroomBatteryEnv.SAVE_DEBUG_DIR + "battery_model.json", "w") as batt_file:
                        json.dump(battery_dict, batt_file)
                except Exception as e:
                    print("Problemi mentre salvavo batteria")
                    print(e)

        return next_state, reward, end_condition, meta_data

    def reset(self, state=None):
        if state is None:
            init_state = self.env.reset()
            if self.use_logger:
                for i in range(len(self.env.net_profile)):
                    if math.isnan(self.env.net_profile[i]):
                        self.nan_values.debug("The sample {} of the net profile is nan".format(i))
        else:
            raise NotImplementedError("This environment does not allow to set the initial state")
        return init_state

    def get_gym_env(self):
        return self.env

    def seed(self, seed):
        self.env.seed(seed)

