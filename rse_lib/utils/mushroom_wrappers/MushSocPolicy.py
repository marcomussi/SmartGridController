from abc import ABC

from mushroom_rl.policy import Policy

from rse_lib.utils.policies import SocPolicy, FixedPolicy


class MushSocPolicy(Policy, ABC):

    def __init__(self, min_soc, max_soc, soc_idx, charge_var_idx):
        self.policy = SocPolicy(min_soc, max_soc, soc_idx=soc_idx, c_rate_idx=charge_var_idx)

    def draw_action(self, state):
        return self.policy.get_action(state)


class MushFixedAction(Policy, ABC):
    def __init__(self, action):
        self.policy = FixedPolicy(action)

    def draw_action(self, state):
        return self.policy.get_action(state)
