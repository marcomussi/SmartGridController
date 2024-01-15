import argparse
import os
import subprocess
import time

if __name__ == "__main__":
    """
    This script is used to automate the testing procedure. The idea is to launch the evaluation of the
    agent, the evaluation of the intermediate agents [optional], the evaluation of the kpis and the fixed actions 
    [optional]
    """
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--exp-names", nargs="+", type=str,
                        help="Name of the experiment that will be tested. It will search directly for the directories "
                             "with the name followed by '_0'. ")
    parser.add_argument("--res-base-path", type=str, default="experiment_folder/FQI/results",
                        help="Directory in which the experiments are held.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of concurrent environments tested. Given enough cores, the performance will "
                             "scale linearly with the number of jobs, since all  simulation are independent")
    parser.add_argument("--freq", type=int, default=30, help="The saved metrics are computed every FREQ day")
    parser.add_argument("--years", type=int, default=-1,
                        help="Number of years the simulation will run. Default is 1, therefore the value will be "
                             "read from a configuration file")
    parser.add_argument("--default-bolun",default=False, action="store_true",
                        help="when set, the simulation will ignore the modifications done to the original Bolun Model "
                             "stored in the configuration")
    parser.add_argument("--test-kpis", default=False, action="store_true", help="Tests the kpis on the same "
                                                                                "environments of the agents")
    args = parser.parse_args()

    RESULT_BASE_PATH = args.res_base_path
    exp_names = args.exp_names
    exp_folders = [x + "_0" for x in exp_names]
    jobs = args.jobs
    freq = args.freq
    years = args.years  # number of years simulated
    default_bolun = args.default_bolun
    test_kpis = args.test_kpis
    # evaluation of final agent
    for index in range(len(exp_folders)):
        start = time.time()
        exp_fold = exp_folders[index]
        """
        python3 evaluate_agent.py --exp-folder EXP_FOLDER [--agent-path AGENT_PATH] --jobs JOBS
        --alg FQI --freq FREQ --years YEARS --default-bolun
        """
        command = ["python3", "evaluate_agent.py"]
        # exp-folder
        exp_path = os.path.join(RESULT_BASE_PATH, exp_fold)
        command.extend(["--exp-folder", exp_path])
        # agent path arg is useless if just need to test the final agent
        # --jobs
        command.extend(["--job", str(jobs)])
        # alg, select fqi to use agent
        command.extend(["--alg", "fqi"])
        # frequency of evaluation
        command.extend(["--freq", str(freq)])
        # years
        if years != -1:
            command.extend(["--years", str(years)])
        # default bolun, ignore degradation value
        if default_bolun:
            command.extend(["--default-bolun"])
            print("ATTENZIONE: file di configurazione ignorato!!!!! Verrà usato il degrado con parametri preimpostati")
        else:
            command.extend(["--postfix", "_orig_deg"])
        print(command)
        exit_code = subprocess.run(command)
        print("Exit code was {}".format(exit_code))
        print("Questo comando è durato {}".format(time.time() - start))
        print()

    if test_kpis:
        for index in range(len(exp_folders)):
            """
            python evaluate_kpis.py
            [--exp-folder EXP_FOLDER] [--jobs JOBS] [--freq FREQ]
              [--years YEARS] [--fixed-action] [--n-actions N_ACTIONS]
              [--env-print]

            """
            start = time.time()
            exp_fold = exp_folders[index]
            command = ["python3", "evaluate_kpis.py"]
            # exp-folder
            exp_path = os.path.join(RESULT_BASE_PATH, exp_fold)
            command.extend(["--exp-folder", exp_path])
            # --jobs
            command.extend(["--job", str(jobs)])
            # frequency of evaluation
            command.extend(["--freq", str(freq)])
            # years
            if years != -1:
                command.extend(["--years", str(years)])
