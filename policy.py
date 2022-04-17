import time
import logging

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
    def __init__(self, model, config=None):
        config = {} if config is None else config
        self.model = model
        config["total_timesteps"] = config.get("total_timesteps",10)
        self.model_path = f"ppo_{time.strftime('%m/%d/%Y_%H:%M:%S')}"
        self.config = config
    def train(self):
        logging.info(f"Learning with params: {self.config}")
        self.model.learn(**self.config)
        logging.info(f"Saving to {self.model_path}")
        self.model.save(self.model_path)
    def predict(self, obs, env_):
        return self.model.predict(obs)
    def __call__(self):
        # return self.model... what?
        pass

class RandomPolicy(Policy):
    """Simple dummy policy
    """
    @staticmethod
    def predict(obs, env_):
        """Return a random sample
        """
        return env_.action_space.sample()

def get_policy(name, env, config):
    """Policy instantiator function

    Args:
        name (str): The policy name
    """
    if name == "ppo":
        ppo = PPO("MlpPolicy", env, verbose=1)
        model = SB3Policy(ppo, config)
    elif name == "random":
        model = RandomPolicy()
    else:
        raise ValueError(f"Undefined policy {name}")
    print(f"Using policy: [{name}]")
    return model