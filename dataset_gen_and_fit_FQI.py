import argparse
import os.path
import subprocess
import time


def append_json_string(names_list):
    output_list = []
    for index in range(len(names_list)):
        name = names_list[index]
        if not name.endswith(".json"):
            name = name + ".json"
        output_list.append(name)
    return output_list


if __name__ == "__main__":
    """
    This script is used to automate launching multiple experiments. It's used to create a dataset since 
    its written when degradation parameters are changed, but the creation of the dataset could be skipped in some cases
    """
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--exp-only-fit", nargs="*", default=[],
                        help="Experiments that will only be fitted, without dataset "
                             "generation")
    parser.add_argument("--only-fit-dataset", type=str, help="Dataset path used to fit the experiments in only fit")

    parser.add_argument("--exp-gen-fit", nargs="*", default=[], help="Experiments that first will need a dataset "
                                                                     "generation, than will be fitted")
    parser.add_argument("--json-base-path", nargs="?", type=str, default="experiment_folder/FQI_json/",
                        help="Path where all the json files "
                             "can be found.")
    parser.add_argument("--dataset-base-path", type=str, default="experiment_folder/FQI/sequential_dataset",
                        help="Folder where dataset will be stored")
    parser.add_argument("--save-frequency", type=int, help="Save frequency of FQI agent")
    args = parser.parse_args()
    print(args)
    JSON_BASE_PATH = args.json_base_path
    DATASET_BASE_PATH = args.dataset_base_path
    exp_fit_names = args.exp_only_fit
    only_fit_dataset_path = args.only_fit_dataset
    exp_genfit_names = args.exp_gen_fit
    save_fit_frequency = args.save_frequency
    print("Dataset base path: {}".format(DATASET_BASE_PATH))
    print("Json base path: {}".format(JSON_BASE_PATH))

    create_dataset = True
    # add json to the end
    json_fit_names = append_json_string(exp_fit_names)
    json_genfit_names = append_json_string(exp_genfit_names)
    # create dataset names (the same of the experiments)
    only_fit_dataset_names = [only_fit_dataset_path] * len(exp_fit_names)  # creates a list of repeated elements
    genfit_dataset_names = list(exp_genfit_names)  # deep copy
    """
    Dataset creation command
    python3 generate_dataset.py --exp-path EXP_PATH --target-json-name NAME --dataset-folder-name FOLD_NAME
    This line is executed only if exp_genfit_names is not empty
    """
    for index in range(len(exp_genfit_names)):
        command = ["python3", "generate_dataset.py"]
        # exp path
        json_name = json_genfit_names[index]
        command.append("--exp-path")
        command.append(os.path.join(JSON_BASE_PATH, json_name))
        # target-json-name
        target_json_name = json_genfit_names[index]
        command.append("--target-json-name")
        command.append(target_json_name)
        # dataset folder name
        dataset_fold_name = genfit_dataset_names[index]
        command.append("--dataset-folder-name")
        command.append(dataset_fold_name)
        exit_code = subprocess.run(command)
        print("Exit code was {}\n\n".format(exit_code))
    # fit command
    """
    Fit FQI command
    python3 fit_FQI.py --data-folder DATA_FOLDER --exp-json EXP_JSON --target-json-name T_J_NAME 
    --folder-name RES_FOLD_NAME --save-fit --save-fit-frequency FREQUENCY
    This command will be run for both experiments that needed the dataset generation and for experiment that
    only needed to be fitted
    """
    # aggregate names for fit
    dataset_names = only_fit_dataset_names + genfit_dataset_names  # deep-copied
    experiment_names = exp_fit_names + exp_genfit_names  # deep-copied
    json_names = json_fit_names + json_genfit_names
    for index in range(len(json_names)):
        command = ["python3", "fit_FQI.py"]
        # data folder, where the dataset is put
        dataset_fold_name = dataset_names[index]
        dataset_path = os.path.join(DATASET_BASE_PATH, dataset_fold_name)
        command.extend(["--data-folder", dataset_path])
        # exp json, where the parameters of the experiment are
        json_name = json_names[index]
        json_exp_path = os.path.join(JSON_BASE_PATH, json_name)
        command.extend(["--exp-json", json_exp_path])
        # target json name
        target_json_name = json_names[index]
        command.extend(["--target-json-name", target_json_name])
        # folder-name is the name of the folder in which the results are put
        result_folder_name = experiment_names[index]
        command.extend(["--folder-name", result_folder_name])
        # save fit
        if save_fit_frequency is not None:
            command.extend(["--save-fit"])
            # save fit frequency
            string_frequency = str(save_fit_frequency)
            command.extend(["--save-fit-frequency", string_frequency])
        print(command)
        start = time.time()
        exit_code = subprocess.run(command)
        end_time = time.time()
        print("Exit code was {}. Runneto for {} seconds\n\n".format(exit_code, start-end_time))
    print("FINEEE")
