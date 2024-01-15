from mushroom_rl.utils.callbacks import Callback
from mushroom_rl.utils.dataset import compute_J


class StepCallbackList(Callback):

    def __init__(self):
        self.callbacks = list()

    def register_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, dataset):
        for callback in self.callbacks:
            callback.__call__(dataset)

    def clean(self):
        super().clean()
        for callback in self.callbacks:
            callback.clean()


class SampleJCallback(Callback):

    def __init__(self, frequency, gamma):
        super().__init__()
        self.j_values = list()
        self.iterations = 0
        self.f = frequency
        self.gamma = gamma

    def __call__(self, dataset):
        self._data_list += dataset
        if self.iterations % self.f == 0 and self.iterations != 0:
            j = compute_J(self._data_list, self.gamma)
            self.j_values.append(j)

    def get(self):
        return self.j_values

    def clean(self):
        super()
        self.j_values = list()


class LinearDegradationCallback(Callback):
    """
    This callback is used to keep track of the degradation factor values. These values
    are the exponent of the formula that computes the loss of capacity in the Bolun Model.
    The value saved are NOT the loss of capacity of SoH. Its called every step
    """

    def __init__(self, env, frequency=None):
        """
        Inizialize the degradation factors as a list of list. Every sublist is the degradation of an episode.

        Args:
            env:
        """
        BASE_FREQ = 1

        super().__init__()
        self.env = env
        self.f_cal_list = [[]]
        self.f_cyc_list = [[]]
        self.stream_f_cyc_past_list = [[]]
        self.is_new_episode = False
        self.frequency = BASE_FREQ if frequency is None else frequency
        self.iterations = 0

    def __call__(self, dataset):
        """
        Saves in the corresponding array the degradation factors. When a new episode is detected, a new list is used
        Args:
            dataset: a list that contains the last sample processed by the environment

        Returns:

        """
        if self.iterations % self.frequency == 0:
            if self.is_new_episode:
                # reset the flag
                self.is_new_episode = False
                self.iterations = 0
                self.f_cal_list.append([])
                self.f_cyc_list.append([])
                self.stream_f_cyc_past_list.append([])

            gym_env = self.env.get_gym_env()
            degradation_model = gym_env.get_degradation_model()
            # calendar ageing
            f_cal = degradation_model.get_calendar_deg()
            self.f_cal_list[-1].append(f_cal)
            # cycling ageing
            f_cyc = degradation_model.get_cycling_deg()
            self.f_cyc_list[-1].append(f_cyc)
            # streamflow closed cycles ageing
            stream_deg = degradation_model.get_stream_f_cyc_past()
            self.stream_f_cyc_past_list[-1].append(stream_deg)
            # [0] to access the sample, [5] is the position of the last step flag that tells if an episode has ended
            if dataset[0][5]:
                self.is_new_episode = True


class SoHCallback(Callback):

    BASE_FREQ = 1

    def __init__(self, env, init_soh, frequency=None):
        super().__init__()
        self.env = env
        self.soh_list = [[]]
        self.is_new_episode = False
        self.frequency = SoHCallback.BASE_FREQ if frequency is None else frequency
        self.iterations = 0
        self.init_soh = init_soh
        self.soh_list[-1].append(self.init_soh)

    def __call__(self, dataset):
        if self.iterations % self.frequency == 0:
            if self.is_new_episode:
                # reset the flag
                self.is_new_episode = False
                self.soh_list[-1].append(self.init_soh)
                self.iterations = 0

            gym_env = self.env.get_gym_env()
            battery_model = gym_env.battery_model
            curr_soh = battery_model.get_soh()
            self.soh_list[-1].append(curr_soh)
            if dataset[0][5]:
                self.is_new_episode = True

    def clean(self):
        super().clean()
        self.soh_list = [[]]
        self.is_new_episode = False
        self.iterations = 0
        self.soh_list[-1].append(self.init_soh)