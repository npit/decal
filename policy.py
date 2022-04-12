import stable_baselines3 as sb3

from stable_baselines3 import PPO

class Policy:
    """Base policy class
    """
    def train(self):
        """Training function
        """

class SB3Policy(Policy):
    """Stable baselines 3 policy
    """
    def __init__(self, model):
        self.model = model
    def train(self):
        self.model.learn(total_timesteps=10000)
    def __call__(self):
        # return self.model... what?
        pass

class RandomPolicy(Policy):
    """Simple dummy policy
    """
    @staticmethod
    def __call__(env_):
        """Return a random sample
        """
        return env_.action_space.sample()

def get_policy(name, env):
    """Policy instantiator function

    Args:
        name (str): The policy name
    """
    if name == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
    elif name == "random":
        model = RandomPolicy()
    else:
        raise ValueError(f"Undefined policy {name}")
    print(f"Using policy: [{name}]")
    return model