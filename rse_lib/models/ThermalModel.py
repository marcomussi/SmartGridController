
class ThermalModel():

    r_eletr = 5e-3  # electric resistance
    r_term = 0.37  # terminal resistance
    c_term = 1.7e3  # termal capacitance

    def __init__(self):
        pass

    def step_temp(self, current, t_sample, curr_temp, t_env):
        power = current**2 * self.r_eletr
        next_temp = (power*self.r_term*t_sample + curr_temp*self.r_term * self.c_term + t_env * t_sample) / \
               (self.r_term*self.c_term + t_sample)
        return next_temp