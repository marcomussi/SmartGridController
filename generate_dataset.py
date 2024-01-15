import argparse
import json
import os
import pickle
import shutil
import time
from datetime import datetime

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter

from rse_lib.environment.MushroomBatteryEnv import MushroomBatteryEnv

if __name__ == "__main__":
    # the parser expects the presence of a configuration file for the environment
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--exp-path", type=str, help="path of the configuration file of the experiment")
    parser.add_argument("--target-json-name", type=str, help="Name of the configuration file put together with the "
                                                             "dataset")
    parser.add_argument("--dataset-folder-name", type=str, help="Name of the folder in which the dataset is saved")
    args = parser.parse_args()
    json_path_args = args.exp_path
    json_name_arg = args.target_json_name
    dataset_folder_name_arg = args.dataset_folder_name

    # load the configuration file
    with open(json_path_args, "r") as f:
        experiment_params = json.load(f)
    experiment_name = experiment_params["name"]
    SEED = experiment_params["train_seed"]

    # setting paths
    BASE_PATH = os.path.join("experiment_folder", "FQI", "sequential_dataset")
    dataset_folder_name = experiment_name + datetime.now().strftime("%d-%m-%Y-%H:%M") \
        if dataset_folder_name_arg is None else dataset_folder_name_arg
    current_dataset_path = os.path.join(BASE_PATH, dataset_folder_name)
    # handle datasets with same name if already exists a dataset
    if os.path.exists(current_dataset_path):
        n_same_name = len([x for x in os.listdir(BASE_PATH) if x.startswith(dataset_folder_name)])
        dataset_folder_name = "{}_{}".format(dataset_folder_name, n_same_name)
        current_dataset_path = os.path.join(BASE_PATH, dataset_folder_name)
    # create recursively the sets
    if not os.path.exists(current_dataset_path):
        os.makedirs(current_dataset_path)
    logger = Logger("creation_dataset", current_dataset_path, log_console=True, seed=SEED)
    logger.info("The dataset will be put at path {}".format(current_dataset_path))
    # print parameters
    #logger.info("Parameters of the experiment\n{}".format(json.dumps(experiment_params, indent=2)))
    start = time.time()
    logger.info("Launched at {}".format(datetime.now().strftime("%d-%m-%Y-%H:%M")))

    # creation and seeding of the experiment
    mdp = MushroomBatteryEnv(name=experiment_params["name"], seed=SEED,
                             experiment_params=experiment_params["environment"],
                             use_logger=experiment_params["use_logger"])
    logger.info("Environment created")
    mdp.seed(SEED)

    fqi_params = experiment_params["fqi_params"]
    # creation of the random policy, used to generate the dataset for FQI
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # very generic approximator used to create the dataset, the epsgreedy is uniform
    approx_params = dict()
    approx_params["input_shape"] = mdp.info.observation_space.shape
    approx_params["output_shape"] = (1,)
    approx_params["n_actions"] = experiment_params["environment"]["n_actions"]
    approximator = LinearApproximator

    # Agent
    algorithm_params = dict(n_iterations=fqi_params["n_iterations"])
    agent = FQI(mdp.info, pi, approximator,
                approximator_params=approx_params, **algorithm_params)
    # Algorithm

    # creates the dataset using the train parameters, since this will be used on testing parameters
    fqi_decision_params = experiment_params["fqi_decision_params"]
    core = Core(agent, mdp)
    # some folder and file naming
    json_name = json_name_arg if json_name_arg is not None else experiment_name + ".json"
    if json_name.endswith(".json"):
        json_name = json_name + ".json"
    logger.info("Starting creating the dataset.")
    fqi_dataset = core.evaluate(n_episodes=experiment_params["learn_params"]["n_episodes"])
    end_time = time.time()
    logger.info("Dataset created in {}".format(end_time - start))

    # save the dataset as a pickle, so that it can be used after

    pickle_path = os.path.join(current_dataset_path, "sequential_dataset.pkl")
    logger.info("Saving the dataset at the path {}".format(pickle_path))
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(fqi_dataset, pickle_file)
    logger.info("Dataset saved")
    # copy configuration file for the dataset
    destination_path = os.path.join(current_dataset_path, json_name)
    shutil.copy(json_path_args, destination_path)
    logger.info("Copyed json from {} to {}".format(json_path_args, destination_path))
