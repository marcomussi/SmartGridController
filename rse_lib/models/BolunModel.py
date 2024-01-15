import math

import numpy as np
import rainflow

from rse_lib.models.Streamflow import Streamflow


def mean_between(start_end, profile):
    return np.mean(profile[start_end[0]:start_end[1]])


class BolunModel:
    """
    This class implements the degradation model of Xu et Al "Modeling of Lithium-Ion Battery Degradation for Cell Life
    Assessment".
    """
    # the following attributes are the values given by the paper
    ALPHA_SEI = 5.75e-2
    BETA_SEI = 121
    K_DELTA1 = 1.4e5
    K_DELTA2 = -5.01e-1
    K_DELTA3 = -1.23e5
    K_SIGMA = 1.04
    SIGMA_REF = 0.5
    K_TEMP = 6.93e-2
    T_REF = 25
    K_CAL = 4.14e-10
    USE_CALENDAR = True

    def __init__(self, initial_soc, n_samples_reset=None, deg_params=None):
        # set calendar aging attributes
        # f_cyc and f_cal are used for logging purposes
        self.f_cal = 0
        self.f_cyc = 0
        self.soc_mean = 0
        self.temp_mean = 0
        self.t = 0
        # srt cycling aging attributes
        self.start_index = 0
        self.f_cyc_past = 0
        self.f_cyc_curr = 0
        self.curr_cycle_len = 0
        self.soc_mean_curr_cycle = 0
        self.temp_mean_curr_cycle = 0
        self.dod_last_cycle = 0
        self.cycles = []
        # init of streamflow
        self.streamflow = Streamflow(init_value=initial_soc)
        self.temp_means_max_len = 0
        self.temp_mean_arr = None
        self.stream_f_cyc_past = 0
        self.old_soc = initial_soc
        if n_samples_reset is not None:
            self.n_samples_reset = n_samples_reset
        # if model_parameters is None, create an empty dictionary, in this way there is no duplicated code
        if deg_params is None:
            deg_params = dict()
        # for each attribute, the default value is set unless is present in the deg params
        self.alpha_sei = BolunModel.ALPHA_SEI if "alpha_sei" not in deg_params else deg_params["alpha_sei"]
        self.beta_sei = BolunModel.BETA_SEI if "beta_sei" not in deg_params else deg_params["beta_sei"]
        self.k_delta1 = BolunModel.K_DELTA1 if "k_delta1" not in deg_params else deg_params["k_delta1"]
        self.k_delta2 = BolunModel.K_DELTA2 if "k_delta2" not in deg_params else deg_params["k_delta2"]
        self.k_delta3 = BolunModel.K_DELTA3 if "k_delta3" not in deg_params else deg_params["k_delta3"]
        self.k_sigma = BolunModel.K_SIGMA if "k_sigma" not in deg_params else deg_params["k_sigma"]
        self.sigma_ref = BolunModel.SIGMA_REF if "sigma_ref" not in deg_params else deg_params["sigma_ref"]
        self.k_temp = BolunModel.K_TEMP if "k_temp" not in deg_params else deg_params["k_temp"]
        self.t_ref = BolunModel.T_REF if "t_ref" not in deg_params else deg_params["t_ref"]
        self.k_cal = BolunModel.K_CAL if "k_cal" not in deg_params else deg_params["k_cal"]
        self.use_cal = BolunModel.USE_CALENDAR if "use_cal" not in deg_params else deg_params["use_cal"]

    def compute_fastflow_degradation(self, soc, temp, delta_dod, t, curr_iter, changed_dir):
        # calendar aging
        self.t = t
        self.soc_mean = (self.soc_mean * curr_iter + soc) / (curr_iter + 1)
        self.temp_mean = (self.temp_mean * curr_iter + temp) / (curr_iter + 1)
        f_cal = self.compute_f_calendar(self.t, self.soc_mean, self.temp_mean)
        # cycle
        self.fast_cycle_counting(soc, temp, delta_dod, curr_iter, changed_dir)
        f_d = f_cal + self.f_cyc_past + self.f_cyc_curr
        bat_deg = self.compute_degradation(f_d)
        return bat_deg

    def compute_rainflow_degradation(self, soc, temp, t, curr_iter, soc_profile, temp_profile):
        # calendar
        self.t = t
        self.soc_mean = (self.soc_mean * curr_iter + soc) / (curr_iter + 1)
        self.temp_mean = (self.temp_mean * curr_iter + temp) / (curr_iter + 1)
        f_cal = self.compute_f_calendar(self.t, self.soc_mean, self.temp_mean)

        rain_gen = rainflow.extract_cycles(soc_profile)
        total_f_cycle = 0
        for cycle_range, mean, cycle_type, i_start, i_end in rain_gen:
            temp_f_cycle = self.compute_f_cycle(cycle_type, cycle_range,
                                                np.mean(soc_profile[i_start:i_end]),
                                                np.mean(temp_profile[i_start:i_end]))
            total_f_cycle += temp_f_cycle
        f_d = f_cal + total_f_cycle
        batt_deg = self.compute_degradation(f_d)
        return batt_deg

    def compute_streamflow_degradation(self, soc, temp, t, curr_iter, charging, temp_profile, expected_end=None):
        """
        This algorithm uses our online approximate implementation of the rainflow algorithm. It has very good performance,
        but after 525 000 samples the algorithm starts to slowdown. In order to get the same performance the streamflow alg
        can be reset.
        Args:
            soc: current value of soc
            temp: current value of temperature
            t: time elapsed from the start of the life of the battery
            curr_iter: number of iterations done by the environment
            charging: boolean flags that keeps track of the direction of the current
            temp_profile: temperature profile of th battery
            expected_end: expected end, its default value is set to none. This means that the alg will try to give a guess
            if no hints are given

        Returns: Degradation of the battery

        """
        # calendar
        self.t = t
        self.soc_mean = (self.soc_mean * curr_iter + soc) / (curr_iter + 1)
        self.temp_mean = (self.temp_mean * curr_iter + temp) / (curr_iter + 1)
        f_cal = self.compute_f_calendar(self.t, self.soc_mean, self.temp_mean)

        # if the current soc is distant delta_soc from one of the soc limits, the expected end
        # is expected to be in between
        if expected_end is None:
            alpha = 0.5
            if charging:
                expected_end = alpha * (1 - soc)
            else:
                expected_end = alpha * soc

        if curr_iter % self.n_samples_reset == 0:
            # TODO save old opened cycle as closed and add them to self.stream_f_cyc_past.
            self.streamflow = Streamflow(self.old_soc)

        soc_means, ranges, n_samples_arr, is_valid, _, start_indexes, temp_means, to_invalid = \
            self.streamflow.update(soc, expected_end, second_signal_value=temp)
        # need to do this in order to avoid DividedByZero exception
        ranges[ranges == 0] += 1e-6
        cycle_types = 0.5
        valid_n_cycles = len(soc_means[is_valid])

        # valid_sample_arr = n_samples_arr[is_valid]
        # start_ends_valid = np.stack((valid_starts, valid_starts+valid_sample_arr)).transpose()
        # axis=1 means we are considering rows
        # valid_temp_means = np.apply_along_axis(mean_between, 1, start_ends_valid, temp_profile)
        valid_temp_means = temp_means[is_valid]
        valid_f_cyc_arr = self.compute_f_cycle(cycle_types, ranges[is_valid], soc_means[is_valid], valid_temp_means)
        f_cyc = np.sum(valid_f_cyc_arr)
        # to invalid part
        if len(to_invalid) != 0:
            invalid_soc_means = soc_means[to_invalid]
            invalid_ranges = ranges[to_invalid]
            invalid_sample_arr = n_samples_arr[to_invalid]
            invalid_start_indexes = start_indexes[to_invalid]
            invalid_n_cycles = len(invalid_soc_means)
            invalid_temp_means = temp_means[to_invalid]
            f_cyc_invalid = np.sum(
                self.compute_f_cycle(cycle_types, invalid_ranges, invalid_soc_means, invalid_temp_means))
            self.stream_f_cyc_past += f_cyc_invalid

        calendar_factor = 1
        if not self.use_cal:
            calendar_factor = 0
        f_d = calendar_factor * f_cal + f_cyc + self.stream_f_cyc_past
        # set attributes used for logging
        self.f_cal = f_cal
        self.f_cyc = f_cyc + self.stream_f_cyc_past
        batt_deg = self.compute_degradation(f_d)
        return batt_deg

    def compute_temp_means(self, n_cycles, n_samples_arr, start_indexes, temp_profile):
        # expand temp means if i need bigger arrays
        if n_cycles > self.temp_means_max_len:
            self.temp_mean_arr = np.ones(n_cycles)
            self.temp_means_max_len = n_cycles

        for i in range(n_cycles):
            start_index = start_indexes[i]
            end_index = start_index + n_samples_arr[i]
            self.temp_mean_arr[i] = np.mean(temp_profile[start_index:end_index])

        return self.temp_mean_arr[0:n_cycles]

    def compute_f_calendar(self, time, soc_mean, temp_mean):
        return BolunModel.f_cal_function(
            time, soc_mean, temp_mean, k_cal=self.k_cal, k_sigma=self.k_sigma, sigma_ref=self.sigma_ref,
            k_temp=self.k_temp, t_ref=self.t_ref)

    def fast_cycle_counting(self, soc, temp, delta_dod, curr_iter, changed_dir):
        # without this line, fast_cycle_counting has some problem handling constant profiles
        if -1e-6 < delta_dod < 1e-6:
            delta_dod = 1.5e-6
        if changed_dir:
            self.f_cyc_past += self.f_cyc_curr
            self.soc_mean_curr_cycle = soc
            self.temp_mean_curr_cycle = temp
            self.dod_last_cycle = delta_dod
            self.dod_last_cycle = clamp(0, self.dod_last_cycle, 1)
            self.f_cyc_curr = self.compute_f_cycle(0.5, self.dod_last_cycle, self.soc_mean_curr_cycle,
                                                   self.temp_mean_curr_cycle)
            self.curr_cycle_len = 1
        else:
            self.soc_mean_curr_cycle = (self.soc_mean_curr_cycle * self.curr_cycle_len + soc) / (
                    self.curr_cycle_len + 1)
            # (x *n + y) / (n +1)
            self.temp_mean_curr_cycle = (self.temp_mean_curr_cycle * self.curr_cycle_len + temp) / (
                    self.curr_cycle_len + 1)
            self.curr_cycle_len += 1
            self.dod_last_cycle += delta_dod
            self.dod_last_cycle = clamp(0, self.dod_last_cycle, 1)
            f_cyc = self.compute_f_cycle(0.5, self.dod_last_cycle, self.soc_mean_curr_cycle, self.temp_mean_curr_cycle)
            self.f_cyc_curr = f_cyc
            assert 0 <= self.dod_last_cycle <= 1

    def compute_f_cycle(self, cycle_type, dod, mean_soc, mean_temp):
        """
        Computes the degradation caused by cycling on the battery
        Args:
            cycle_type: array whose value can be 0.5 or 1. this array tells if a cycle is a half or a full cycle
            dod: array that contains the depth of discharge for each cycle
            mean_soc: array that contains the mean soc value of each cycle
            mean_temp: array that contains the mean temp value of each cycle

        Returns: the cycling contribution to the degradation

        """
        return BolunModel.f_cycle_function(
            cycle_type, dod, mean_soc, mean_temp, k_delta1=self.k_delta1, k_delta2=self.k_delta2,
            k_delta3=self.k_delta3, k_sigma=self.k_sigma, sigma_ref=self.sigma_ref, k_temp=self.k_temp,
            t_ref=self.t_ref)

    def compute_degradation(self, f_d):
        bat_deg = BolunModel.degradation_function(f_d, alpha_sei=self.alpha_sei, beta_sei=self.beta_sei)
        assert 0 <= bat_deg <= 1, "The degradation is not between 0 and 1, batt_deg = {}".format(bat_deg)
        return bat_deg

    def get_calendar_deg(self):
        return self.f_cal

    def get_cycling_deg(self):
        """

        Returns: The cycling degradation factor.

        """
        # no need to sum the self.stream_f_cyc_past because it's already taken into account
        return self.f_cyc

    def get_fd(self):
        return self.get_calendar_deg() + self.get_cycling_deg()

    def get_stream_f_cyc_past(self):
        return self.stream_f_cyc_past

    @classmethod
    def degradation_function(cls, f_d, alpha_sei=ALPHA_SEI, beta_sei=BETA_SEI):
        bat_deg = 1 - alpha_sei * math.exp(-beta_sei * f_d) - \
                  (1 - alpha_sei) * math.exp(-f_d)
        return bat_deg

    @classmethod
    def f_cycle_function(cls, cycle_type, dod, mean_soc, mean_temp, k_delta1=K_DELTA1, k_delta2=K_DELTA2,
                         k_delta3=K_DELTA3,
                         k_sigma=K_SIGMA, sigma_ref=SIGMA_REF, k_temp=K_TEMP, t_ref=T_REF):
        s_dod_cycle = (k_delta1 * dod ** k_delta2 + k_delta3) ** -1
        s_sigma_cycle = np.exp(k_sigma * (mean_soc - sigma_ref))
        s_temp_cycle = np.exp(k_temp * (mean_temp - t_ref) * (t_ref / mean_temp))
        return cycle_type * s_dod_cycle * s_sigma_cycle * s_temp_cycle

    @classmethod
    def f_cal_function(cls, time, soc_mean, temp_mean, k_cal=K_CAL, k_sigma=K_SIGMA, sigma_ref=SIGMA_REF, k_temp=K_TEMP,
                       t_ref=T_REF):
        s_t_cal = k_cal * time
        s_sigma_cal = math.exp(k_sigma * (soc_mean - sigma_ref))
        s_temp_cal = math.exp(k_temp * (temp_mean - t_ref) * (t_ref / temp_mean))
        return s_t_cal * s_sigma_cal * s_temp_cal

    @classmethod
    def bisection_fd(cls, soh_loss, tol=1e-9, lower=0, upper=100):
        """
        This method is used to compute the exponent of the degradation function for a given loss of capacity.
        For example, in order to have capacity of 90% the loss of SoH is 10%, and the algorithm finds the value of
        fd such that the degradation is 10%.
        Args:
            soh_loss: loss of soh
            tol: tolerance value, the bisection algorithm stops when this threshold is surpassed
            lower: lower starting bound of the bisection method
            upper: upper starting bound of the bisection method

        Returns:

        """
        delta = 1
        c = (upper + lower) / 2
        while delta > tol:
            diff = soh_loss - cls.degradation_function(c)
            if diff > 0:
                lower = c
            else:
                upper = c
            c = (upper + lower) / 2
            delta = abs(diff)
        return c



def clamp(minimum, x, maximum):
    """
    Clamps the value x between minimun and maximum
    Args:
        minimum:
        x:
        maximum:

    Returns:

    """
    return max(minimum, min(x, maximum))
