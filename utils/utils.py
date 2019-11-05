import numpy as np
import matplotlib.pyplot as plt

class Util:
    """get only one random number
    """
    random_sets = {}
    # here, we use @staticmethod to recall methods in class Util
    @staticmethod
    def unique_id(group='default'):
        """This function is to find a random tensor id 
        Args:
            group: string, default value 'default'
        """
        # inti random id
        random_id = np.random.randint(0, 1000000)
        # make sure ranodm_id is not in random_sets
        if group not in Util.random_sets:
            Util.random_sets[group] = set([random_id])
            return random_id
        # make loop to find one random_id non repeated
        while random_id in Util.random_sets[group]:
            random_id = np.random.randint(0, 1000000)
        # add found random_id into set
        Util.random_sets[group].add(random_id)
        return random_id

    @staticmethod
    def clear():
        """clear random set
        """
        Util.random_sets = {}






