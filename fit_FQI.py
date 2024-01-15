import argparse
import json
import os
import pickle
import time
from datetime import datetime

import xgboost as xgb
from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Logger
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from sklearn.ensemble import ExtraTreesRegressor

from rse_lib.environment.MushroomBatteryEnv import MushroomBatteryEnv
from rse_lib.utils.support_agents import FqiSaveIteration

if __name__ == "__main__":
    # the parser expects the presence of a configuration file for the environment
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--data-folder", type=str, help="folder where the dataset is stored")
    parser.add_argument("--exp-json", type=str, help="path to the json configuration. This json is used to set the parameters of the gym environment")
    parser.add_argument("--jobs", type=int, help="number of jobs used to build the extra trees in the case an extratree regressor is used")
    parser.add_argument("--target-json-name", type=str, help="Name of the json that will be saved in the result folder")
    parser.add_argument("--approx", type=str,
                        help="Aprroximator used. The current regressors supported are \"extratrees\" and \"xgboost\"",
                        choices=["extratrees", "xgboost"])
    parser.add_argument("--folder-name", type=str, help="Name of the folder in which the trained agent is stored, together with its configuration")
    parser.add_argument("--regressor-type", type=str, choices=["Q", "action"],
                        help="Type of regressor is used. If Q, only one regressor is built with output "
                             "shape equal to the number of actions, otherwise n_actions regressors are built")
    parser.add_argument("--save-fit", default=False, action="store_true",
                        help="Enables saving the agent while is learning")
    parser.add_argument("--save-fit-frequency", type=int, help="Frequency at which an agent is saved during learning. the --save-fit argument should be set.")
    parser.add_argument("--env-print", default=False, action="store_true")
    args = parser.parse_args()

    fqi_dataset_folder = args.data_folder
    n_jobs = args.jobs
    json_name_arg = args.target_json_name
    configuration_path = args.exp_json
    approx_type_arg = args.approx
    folder_name = args.folder_name
    regressor_type_args = args.regressor_type
    save_fit_arg = args.save_fit
    save_fit_frequency_arg = args.save_fit_frequency
    env_print = args.env_print
    assert (save_fit_arg and save_fit_frequency_arg is not None) or (
                not save_fit_arg and save_fit_frequency_arg is None)
    # load the configuration file
    files = os.listdir(fqi_dataset_folder)
    json_path = configuration_path
    with open(json_path, "r") as f:
        experiment_params = json.load(f)
    experiment_name = experiment_params["name"]
    SEED = experiment_params["train_seed"]

    # CREATION OF FOLDERS
    # result folder
    if folder_name is None:
        result_folder_name = experiment_name + datetime.now().strftime("%d_%m_%Y_%H-%M")
    else:
        results_path = os.path.join("experiment_folder", "FQI", "results")
        n_names = len([x for x in results_path if x.startswith(folder_name)])
        result_folder_name = "{}_{}".format(folder_name, n_names)
    RESULT_FOLDER = os.path.join("experiment_folder", "FQI", "results", result_folder_name)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    # agent save path
    AGENT_SAVE_PATH = os.path.join(RESULT_FOLDER, "agent.msh")
    # creation of logger and its folder
    logger = Logger("fit", RESULT_FOLDER, log_console=True, seed=SEED)
    logger.info("Created result folder. It's path is {}.".format(RESULT_FOLDER))
    # json config save path
    json_name = json_name_arg if json_name_arg is not None else experiment_name + ".json"
    if not json_name.endswith(".json"):
        json_name += ".json"
    logger.info("Setted json name.")
    JSON_CONF_SAVE_PATH = os.path.join(RESULT_FOLDER, json_name)
    with open(JSON_CONF_SAVE_PATH, "w") as json_conf_file:
        json.dump(experiment_params, json_conf_file, indent=2)
    logger.info("Saved json configuration before running experiment.")
    log_path = RESULT_FOLDER
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # print parameters
    if env_print:
        logger.info("Parameters of the experiment\n{}".format(json.dumps(experiment_params, indent=1)))
    logger.info("Experiment launched at {}.".format(datetime.now().strftime("%d-%m-%Y-%H:%M")))
    # creation and seeding of the experiment
    mdp = MushroomBatteryEnv(name=experiment_params["name"], seed=SEED,
                             experiment_params=experiment_params["environment"],
                             use_logger=experiment_params["use_logger"])
    mdp.seed(SEED)
    # creating approximator
    fqi_params = experiment_params["fqi_params"]
    regressor_type = regressor_type_args if regressor_type_args is not None else fqi_params["regressor_type"]
    if regressor_type_args is not None:
        logger.warning("Ignoring regressor type of congiguration file."
                       " Passing from {} to {}".format(fqi_params["regressor_type"], regressor_type))
        experiment_params["fqi_params"]["regressor_type"] = regressor_type

    # creation of the random policy, used just to create the agent
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)
    # load approximator. The type is the one in the configuration file if not specified with command line
    old_approx_type = experiment_params["approximator_type"]
    if approx_type_arg is not None:
        approx_type = approx_type_arg
        if approx_type in ["xgboost", "extratrees"]:
            experiment_params["approximator_type"] = approx_type
        else:
            raise ValueError("Wrong approximator type")
        logger.warning("Ignored approcimator type. Changed from {} to {}".format(old_approx_type, approx_type))
    else:
        approx_type = experiment_params["approximator_type"]

    approximator = None
    approximator_params = None
    if approx_type == "extratrees":
        extratrees_params = experiment_params["approx_extratrees"]
        approx_jobs = n_jobs if n_jobs is not None else extratrees_params["n_jobs"]
        approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                                   n_actions=mdp.info.action_space.n,
                                   n_estimators=extratrees_params["n_estimators"],
                                   min_samples_split=extratrees_params["min_samples_split"],
                                   min_samples_leaf=extratrees_params["min_samples_leaf"],
                                   n_jobs=approx_jobs)
        approximator = ExtraTreesRegressor
        logger.info("The regressor is an extratree.")
    elif approx_type == "xcboost" or approx_type == "xgboost":
        xgboost_params = experiment_params["approx_xcboost"]
        try:
            xgb.set_config(verbosity=xgboost_params.pop("verbosity"))
        except Exception as e:
            logger.warning("It was not possible to pop verbosity parameters. It will be set to 0. Error message: {}".format(e))
            xgb.set_config(verbosity=0)

        approximator = xgb.XGBRegressor
        approximator_params = xgboost_params
        approximator_params["input_shape"] = mdp.info.observation_space.shape
        n_actions = mdp.info.action_space.n
        if regressor_type.lower() == "q":
            # this will create a q regressor
            output_shape = (n_actions,)
            logger.info("The agent will approximate the Q-function with a QRegressor.")
        elif regressor_type.lower() == "action":
            # this will create an action regressor ( n_actions regressor, slower )
            output_shape = (1,)
            logger.info("The agent will approximate the Q-function with an ActionRegressor.")
        else:
            raise NotImplementedError("The {} regressor type is not defined".format(regressor_type))
        approximator_params["n_actions"] = n_actions
        approximator_params["output_shape"] = output_shape
        logger.info("The regressor is xgboost")

    # Agent
    algorithm_params = dict(n_iterations=fqi_params["n_iterations"])

    if save_fit_arg:
        agent = FqiSaveIteration(mdp.info, pi, approximator, approximator_params=approximator_params,
                                 frequency=save_fit_frequency_arg, save_path=RESULT_FOLDER, logger=logger, **algorithm_params)
        logger.info("FQI agent with save every {} iteration at path {} created.".format(
            save_fit_frequency_arg, RESULT_FOLDER))
    else:
        agent = FQI(mdp.info, pi, approximator,
                    approximator_params=approximator_params, **algorithm_params)
        logger.info("FQI agent created.")
    # Algorithm
    pickle_name = [x for x in files if x.endswith(".pkl")][0]
    pickle_path = os.path.join(fqi_dataset_folder, pickle_name)
    logger.info("Loading the dataset at {} .".format(pickle_path))
    with open(pickle_path, "rb") as pickle_file:
        fqi_dataset = pickle.load(pickle_file)
        logger.info("Dataset loaded.")
    logger.info("Fit the agent.")
    start = time.time()
    agent.fit(fqi_dataset)
    end_t = time.time()
    logger.info("Agent fitted. Fitting time: {} seconds.".format(end_t - start))
    logger.info("Saving agent at {} .".format(AGENT_SAVE_PATH))
    agent.save(AGENT_SAVE_PATH)
    logger.info("Agent saved.")
    # save the experiment configuration file
    with open(JSON_CONF_SAVE_PATH, "w") as json_conf_file:
        json.dump(experiment_params, json_conf_file, indent=2)
    logger.info("Dumped the configuration file in the result folder")
