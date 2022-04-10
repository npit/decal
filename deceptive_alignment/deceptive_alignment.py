from gym import Env
from gym.spaces import Discrete

"""Environment module
"""

class DeceptiveAlignment(Env):
    def __init__(self, num_choices: int=2):
        """Constructor
        """
        self.action_space = Discrete(num_choices)
        self.observation_space = Discrete(2)

    def step(self, action):
        """Agent step function

        Args:
            action (gym.spaces): Current action

        Returns:
            state, reward, done, info: Stepping return tuple
        """
        state = 1
    
        if action == 2:
            reward = 1
        else:
            reward = -1
            
        done = True
        info = {}
        return state, reward, done, info

    def reset(self):
        state = 0
        return state
        


    def base_reward(self, observation, action):
        """
        Computes reward of the base optimizer, wrt. input observations and actions
        """

    def mesa_reward(self, observation, action):

