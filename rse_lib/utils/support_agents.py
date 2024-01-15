import os
import time

import numpy as np
from mushroom_rl.algorithms.value import FQI
from mushroom_rl.utils.dataset import parse_dataset
from tqdm import trange


class FqiSaveIteration(FQI):
    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False, frequency=None, save_path=None, logger=None):
        super().__init__(mdp_info=mdp_info, policy=policy, approximator=approximator, n_iterations=n_iterations,
                         approximator_params=approximator_params, fit_params=fit_params, quiet=quiet)
        self.frequency = frequency
        self.save_path = save_path
        assert (self.frequency is None and self.save_path is None) or \
               (self.frequency is not None and self.save_path is not None)
        self.logger = logger

    def log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def fit(self, x):
        start = time.time()
        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        iteration_index = 0
        for _ in trange(self._n_iterations(), dynamic_ncols=True, disable=self._quiet, leave=False):
            if self._target is None:
                self._target = reward
            else:
                q = self.approximator.predict(next_state)
                if np.any(absorbing):
                    q *= 1 - absorbing.reshape(-1, 1)

                max_q = np.max(q, axis=1)
                self._target = reward + self.mdp_info.gamma * max_q

            self.approximator.fit(state, action, self._target, **self._fit_params)
            if self.frequency is not None:
                if iteration_index != 0 and iteration_index % self.frequency == 0:
                    agent_save_name = "agent_iter_{}.msh".format(iteration_index)
                    agent_save_path = os.path.join(self.save_path, agent_save_name)
                    self.save(agent_save_path)
                    elapsed = time.time() - start
                    self.log("Iteration {}, time elapsed {} s. Save agent at {}".format(
                        iteration_index, elapsed, agent_save_path))

            iteration_index += 1
