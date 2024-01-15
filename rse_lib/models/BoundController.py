
class BoundController:
    """
    This controller is used to protect the battery from too high powers that will not respect the battery constraint
    """

    MIN = 0
    MAX = 1
    TOLERANCE = 1e-6

    def __init__(self, lower_bound, upper_bound):
        self.MIN = lower_bound
        self.MAX = upper_bound
        self.tolerance = BoundController.TOLERANCE

    def check_action(self, action, action_min=None, action_max=None):
        correct_min = self.MIN if action_min is None else action_min
        correct_max = self.MAX if action_max is None else action_max
        if correct_min < action < correct_max:
            to_return = action
        elif action <= correct_min:
            to_return = correct_min
        elif action >= correct_max:
            to_return = correct_max - self.tolerance
        else:
            raise ValueError("Some weird exception with the action happened")
        return to_return
