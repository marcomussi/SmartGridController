# A Reinforcement Learning Controller Optimizing Costs and Battery State of Health in Smart Grids

## Marco Mussi, Luigi Pellegrino, Oscar Francesco Pindaro, Marcello Restelli and Francesco Trovò

This is a library for Lithium-Ion Battery simulation. It models and simulates a domestic environment composed of a house that consumes electric energy and a photo-voltaic panel that produces energy. The domestic environment is linked to a controller that decides how much of the net power coming from the domestic environment will be directed on an accumulation system and how much will be bought/sold on the electric grid.
The simulator has been written following the standard framework [Gym](https://gym.openai.com/) from [OpenAI](https://openai.com/). The repository contains a wrapper for the library [MushroomRL](https://mushroomrl.readthedocs.io/en/latest/), used to design and test RL agents.

### Installation
Use [Conda](https://docs.conda.io/en/latest/) to install all the dependencies of RLithium using the provided configuration file _requirements.yml_.

```sh
conda env create -f requirements.yml --name rlithium_env
conda activate rlithium_env
```

This library requires the dependencies of [Gym](https://gym.openai.com/docs/) and [MushroomRL](https://github.com/MushroomRL/mushroom-rl). Visit their installation page for instructions on external dependencies.
On some systems, it may be necessary to set an environment variable for the _open-mpi_ dependency.

```sh
MPICC=/path/to/your/mpicc/executable
```

### Repository Structure
The repository is structured in the following way:
* **current_directory**: contains the scripts used to train and test an agent using the Fitted Q-iteration algorithm
* **./rse_lib**: contains the source code of the repository and the definition of the Gym and MushroomRL environments
* **./data**: contains data used to perform the simulations
* **./notebooks**: contains some Python notebooks with examples and visualization
* **./experiment_folder**: contains the configuration files and the results of the training and testing phase

### Usage
The gym environment is created from the configuration file, that holds all the information regarding simulation parameters and agent parameters. This file is in a format JSON and can be easily edited for different simulations.
To train an agent, the following command should be run.

```sh
python dataset_gen_and_fit_FQI.py --exp-gen-fit CONF_NAME --json-base-path=BASE_PATH --save-frequency=FREQUENCY
```

This command will launch a training with the parameters described in the file held in BASE_PATH/CONF_NAME. The script will create a dataset for the Fitted Q-Iteration Algorithm, and then fit the agent for the number of steps indicated in the configuration file. During the fit, the agent saves every FREQUENCY steps.
The results are saved in ./experiment_folder/FQI/results/CONF_NAME_0.

To test the agent, the following commands have to be run:

```sh
python multiple_exp_testing_FQI.py --exp-names NAME1 NAME2 --jobs=N_JOBS --freq=FREQ
python evaluate_kpis.py --exp-folder PATH_TO_EXP --jobs=JOBS
```

The first command will test all the agents in the folder with prefix NAME1 and NAME2. The number of jobs is used for parallel evaluation. The speed-up in testing performance is linear if there are enough cores. The --freq flag tells how often save the evaluation result so that an analysis on KPI histories can be done.
The second command tests the baselines **SoC20-80**, **Only_grid** and **Only_battery**. In this script, the whole path to the result folder has to be provided.




