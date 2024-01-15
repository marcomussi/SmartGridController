import os

import numpy as np
import pandas as pd


class DataGenerator:
    """
    Generates the profile of the photo-voltaic panel and of the house. The data is generated from real-world data
    """
    PV_PATH = "data/PV_1year/"
    LOAD_PATH = "data/load/Carico.csv"

    WINDOW_SIZE = 3
    T_SAMPLE_DATA = 60 * 60  # seconds

    def __init__(self, pv_path=None, load_path=None, random_start=False, seed=None):
        self.pv_path = pv_path if pv_path is not None else DataGenerator.PV_PATH
        self.load_path = load_path if load_path is not None else DataGenerator.LOAD_PATH

        self.pv_dataframes = []
        self.load_pv_profiles()

        self.load_profiles = pd.DataFrame()
        self.load_load_profiles()

        self.random_start = False

        self.curr_pv_profile = None
        self.curr_load_profile = None

        self.np_rand_gen = np.random.default_rng(seed)

    def load_pv_profiles(self):
        pv_files = os.listdir(self.pv_path)
        pv_files = [x for x in pv_files if x[-3:] == "csv"]
        pv_files.sort()
        for fileName in pv_files:
            df = pd.read_csv(os.path.join(self.pv_path, fileName))
            self.pv_dataframes.append(df)

    def load_load_profiles(self):
        self.load_profiles = pd.read_csv(self.load_path)

    # The cration of the profile assumes that the original data is sampled by the hour. The output is in Wh
    def create_profile(self, years, t_sample):
        # pv_profile is a profile normalized between 0 and 1. An additional year is created
        # a piece of the first year is cut and the complementary of the last is also cut
        # In this way, there's the same number of sample, but with different starts.
        self.curr_pv_profile = None
        self.curr_load_profile = None
        if not self.random_start:
            pv_profile_norm = self.create_pv_profile(years)
            load_profile = self.create_load_profile(years)
        else:
            pv_profile_norm = self.create_pv_profile(years + 1)
            load_profile = self.create_load_profile(years + 1)
            # random sample in a year
            a = 0
            seconds_one_day = 60 * 60 * 24
            sample_one_day = seconds_one_day // DataGenerator.T_SAMPLE_DATA
            sample_one_year = sample_one_day * 365  # just to be sure with leap years, could also be 365 days
            r = self.np_rand_gen.integers(0, sample_one_year - 1)
            pv_profile_norm = pv_profile_norm[r: -(sample_one_year - r)]
            load_profile = load_profile[r: -(sample_one_year - r)]

        max_consumption = np.amax(load_profile)
        pv_profile = max_consumption * pv_profile_norm
        # everything is now expressed in W
        min_len = min(len(load_profile), len(pv_profile))  # may have different length if leap years
        battery_profile = load_profile[0:min_len] - pv_profile[0:min_len]
        battery_profile = self.upsample(battery_profile, t_sample)
        # save current profile
        self.curr_pv_profile = pv_profile[0:min_len]
        self.curr_load_profile = load_profile[0:min_len]
        return battery_profile, max_consumption

    def create_pv_profile(self, years):
        # creation of the pv_profile
        profiles = []
        for year_index in range(years):
            random_index = self.np_rand_gen.integers(0, len(self.pv_dataframes) - 1)
            chosen_dataframe = self.pv_dataframes[random_index]
            power_profile = chosen_dataframe["Power"]
            profile = self.shuffle_days(power_profile)
            profile = DataGenerator.cut_leap_year(profile)
            profiles.append(profile)
        pv_profile = np.concatenate(profiles)
        return pv_profile

    def create_load_profile(self, years):
        load_years = []
        # the upper bound can be also returned, therefore -1
        selected_profile = self.np_rand_gen.integers(0, len(self.load_profiles.columns) - 1)
        for i in range(years):
            shuffled_profile = self.shuffle_days(self.load_profiles[str(selected_profile)])
            shuffled_profile = DataGenerator.cut_leap_year(shuffled_profile)
            load_years.append(shuffled_profile)
        load_profile = np.concatenate(load_years)
        # the profile is expressed in kwH. Since the sampling is done in 1 hour, 1 kWh => 1kW. Then, need to
        # conversion from kW to W, by multiplying by 1000
        load_profile = 1000 * load_profile
        return load_profile

    def shuffle_days(self, profile):
        """
        For each day, a time window is considered. Then, a day in that time window
        is sampled and set as the current day. This sampling is done with reimmission.
        This method expects that the profile starts at the first of January of any year
        and ends at the 31st of December of any year.
        param profile: real power profile used to generate a synthetic profile
        :return:
        """
        gen_profile = np.zeros((len(profile)))
        for day in range(0, len(profile)):
            start_day = max(0, day - self.WINDOW_SIZE)
            end_day = min(day + self.WINDOW_SIZE, len(profile) - 1)
            selected_day = self.np_rand_gen.integers(start_day, end_day)
            start_index = selected_day * 24
            end_index = selected_day * 24 + 24  # last index is not considered
            gen_profile[start_index:end_index] = profile[start_index:end_index]
        return gen_profile

    @classmethod
    def cut_leap_year(cls, profile):
        assert len(profile) >= 24 * 30 * 12
        return profile[0:24 * 30 * 12]

    def upsample(self, profile, t_sample):
        factor = self.T_SAMPLE_DATA / t_sample
        return profile.repeat(factor, axis=0)

    def seed(self, seed):
        self.np_rand_gen = np.random.default_rng(seed)


class DelayedDataGenerator(DataGenerator):
    """
    Adds a delay in the house profile, in order to control when the battery should be active
    """

    def __init__(self, pv_path=None, load_path=None, random_start=False, seed=None, hours_shift=0):
        super().__init__(pv_path, load_path, random_start, seed)
        self.hours_shift = hours_shift

    def create_profile(self, years, t_sample):
        self.curr_pv_profile = None
        self.curr_load_profile = None
        if not self.random_start:
            pv_profile_norm = self.create_pv_profile(years)
            load_profile = self.create_load_profile(years)
        else:
            raise NotImplementedError("Not Implemented the random start for this class")
        load_profile = DelayedDataGenerator.shift_profile(load_profile, self.hours_shift)
        max_consumption = np.amax(load_profile)
        pv_profile = max_consumption * pv_profile_norm
        # everything is now expressed in W
        min_len = min(len(load_profile), len(pv_profile))  # may have different length if leap years
        battery_profile = load_profile[0:min_len] - pv_profile[0:min_len]
        battery_profile = self.upsample(battery_profile, t_sample)
        # save current profile
        self.curr_pv_profile = pv_profile[0:min_len]
        self.curr_load_profile = load_profile[0:min_len]
        return battery_profile, max_consumption

    @classmethod
    def shift_profile(cls, profile, shift):
        return np.roll(profile, shift)
