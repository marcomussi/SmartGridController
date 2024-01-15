import numpy as np


class Streamflow:
    """
    Implementation of ouw cycle counting algorithm, that is able to perform in an online manner without considering
    every new sample the whole soc and temperature history. It's inspired to the rainflow cycle counting algorithm.
    """

    def __init__(self, init_value=0, subsample=False, interpolate='linear',
                 expected_cycle_num=500, cycle_num_increment=500):

        self.mean_values = np.zeros(expected_cycle_num)
        # second signal mean values
        self.second_signal_means = np.zeros(expected_cycle_num)
        self.range_size = np.zeros(expected_cycle_num)
        self.min_max_vals = np.zeros((expected_cycle_num, 2))
        self.number_of_samples = np.zeros(expected_cycle_num, dtype=int)
        self.start_cycles = np.zeros(expected_cycle_num, dtype=int)
        self.end_cycles = np.zeros(expected_cycle_num, dtype=int)
        # number of samples in the cycle
        self.directions = np.zeros(expected_cycle_num, dtype=int)
        # 1 -> up, 2 -> down
        self.is_valid = np.zeros(expected_cycle_num, dtype=bool)
        # cycles that are waiting to be completed
        self.is_used = np.zeros(expected_cycle_num, dtype=bool)
        # mask to hide cycles allocated for efficiency purpose only
        self.aux_enum = np.linspace(0, expected_cycle_num - 1, expected_cycle_num, dtype=int)
        self.last_value = init_value
        self.actual_cycle = -1
        self.subsample = subsample
        # to implement to manage edge cases
        self.interpolate = interpolate
        # only linear now
        self.cycle_num_increment = cycle_num_increment
        # for performance purpose only
        self.is_init = True
        # iterations
        self.iteration = 0

    def update(self, actual_value, expected_end, second_signal_value=None, return_valid_only=True,
               return_unvalidated_list=True):

        change_direction = False
        # used to obtain information about close cycles
        to_invalid = []
        if self.is_init or (
                self.directions[self.actual_cycle] == 1 and actual_value < self.last_value
        ) or (self.directions[self.actual_cycle] == 2 and actual_value > self.last_value):
            change_direction = True
            self.is_init = False

        if change_direction:

            self.actual_cycle = np.sum(self.is_used)

            if self.actual_cycle >= self.range_size.shape[0]:
                self.expand()

            # enable cycle and set the direction
            self.is_valid[self.actual_cycle] = True
            self.is_used[self.actual_cycle] = True
            if actual_value < self.last_value:
                self.directions[self.actual_cycle] = 2
            else:
                self.directions[self.actual_cycle] = 1

            self.start_cycles[self.actual_cycle] = self.iteration
            min_val, max_val = min(actual_value, self.last_value), max(actual_value, self.last_value)
            self.min_max_vals[self.actual_cycle] = (min_val, max_val)
            self.range_size[self.actual_cycle] = max_val - min_val
            self.mean_values[self.actual_cycle] = actual_value
            if second_signal_value is not None:
                self.second_signal_means[self.actual_cycle] = second_signal_value
            self.number_of_samples[self.actual_cycle] = 1
            # self.mean_values[self.actual_cycle] = 0
            # self.number_of_samples[self.actual_cycle] = 0

        else:  # same direction

            valid_used_correct_direction = self.is_used & self.is_valid \
                                           & (self.directions == self.directions[self.actual_cycle])

            is_up = self.directions[self.actual_cycle] == 1

            up_indexes = (self.min_max_vals[valid_used_correct_direction][:, 1] > self.last_value
                          ) & (self.min_max_vals[valid_used_correct_direction][:, 1] < actual_value)

            down_indexes = (self.min_max_vals[valid_used_correct_direction][:, 0] < self.last_value
                            ) & (self.min_max_vals[valid_used_correct_direction][:, 0] > actual_value)
            indexes = up_indexes if is_up == 1 else down_indexes

            expected_indexes_up = (self.min_max_vals[valid_used_correct_direction][:, 1] > actual_value
                                   ) & (self.min_max_vals[valid_used_correct_direction][:, 1] < expected_end)
            expected_indexes_down = (self.min_max_vals[valid_used_correct_direction][:, 0] < actual_value
                                     ) & (self.min_max_vals[valid_used_correct_direction][:, 0] > expected_end)
            expected_end_indexes = expected_indexes_up if is_up else expected_indexes_down
            min_max_index = 1 if is_up else 0

            arg_function = np.argmax if is_up else np.argmin
            if indexes.any():

                aux = self.aux_enum[valid_used_correct_direction]
                # if something will fall later, invalid all the current falling cycles
                if expected_end_indexes.any():
                    to_invalid = aux[indexes]
                    self.is_valid[to_invalid] = False
                # else, if nothing is falling, find the biggest that is falling
                else:
                    # indexes is boolean, therefore aux[indexes] filtra
                    self.is_valid[self.actual_cycle] = False
                    self.actual_cycle = arg_function(self.min_max_vals[aux[indexes]][:, min_max_index])
                    to_invalid = np.concatenate((aux[indexes][0:self.actual_cycle],
                                                 aux[indexes][self.actual_cycle + 1:len(aux[indexes])]))
                    self.is_valid[to_invalid] = False
                    self.is_valid[self.actual_cycle] = True

            # update of the mean and the ranges
            self.mean_values[self.actual_cycle] = (self.mean_values[
                                                       self.actual_cycle] * self.number_of_samples[self.actual_cycle] +
                                                   actual_value) / (self.number_of_samples[self.actual_cycle] + 1)
            if second_signal_value is not None:
                self.second_signal_means[self.actual_cycle] = \
                    (self.second_signal_means[self.actual_cycle] * self.number_of_samples[self.actual_cycle] +
                     second_signal_value) / (self.number_of_samples[self.actual_cycle] + 1)
            bounds_tuple = self.min_max_vals[self.actual_cycle]
            min_val, max_val = min(bounds_tuple[0], actual_value), max(bounds_tuple[1], actual_value)
            self.min_max_vals[self.actual_cycle] = (min_val, max_val)
            self.range_size[self.actual_cycle] = max_val - min_val
            # TODO potrebbe risolvere non monotonia
            self.number_of_samples[self.actual_cycle] += 1

        valid_and_used = np.logical_and(self.is_used, self.is_valid)
        # self.number_of_samples[valid_and_used] += 1  # TODO this could be a problem
        self.last_value = actual_value
        mask = self.is_used
        self.iteration += 1

        return self.mean_values[mask], self.range_size[mask], self.number_of_samples[mask], self.is_valid[
            mask], change_direction, self.start_cycles[mask], self.second_signal_means[mask], to_invalid

    def expand(self):
        # more cycles than the one pre-allocated, need to allocate new ones
        self.mean_values = np.concatenate(
            (self.mean_values, np.zeros(self.cycle_num_increment)))
        self.second_signal_means = np.concatenate(
            (self.second_signal_means, np.zeros(self.cycle_num_increment)))
        self.range_size = np.concatenate(
            (self.range_size, np.zeros(self.cycle_num_increment)))
        self.min_max_vals = np.vstack(
            (self.min_max_vals, np.zeros((self.cycle_num_increment, 2))))
        self.number_of_samples = np.concatenate(
            (self.number_of_samples, np.zeros(self.cycle_num_increment, dtype=int)))
        self.start_cycles = np.concatenate(
            (self.start_cycles, np.zeros(self.cycle_num_increment, dtype=int))
        )
        self.end_cycles = np.concatenate(
            (self.end_cycles, np.zeros(self.cycle_num_increment, dtype=int))
        )
        self.directions = np.concatenate(
            (self.directions, np.zeros(self.cycle_num_increment, dtype=int)))
        self.is_valid = np.concatenate(
            (self.is_valid, np.zeros(self.cycle_num_increment, dtype=bool)))
        self.is_used = np.concatenate(
            (self.is_used, np.zeros(self.cycle_num_increment, dtype=bool)))
        prev_len = self.aux_enum.shape[0]
        self.aux_enum = np.concatenate(
            (self.aux_enum, np.linspace(prev_len, prev_len +
                                        self.cycle_num_increment - 1,
                                        self.cycle_num_increment, dtype=int)))
