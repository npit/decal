"""Module for experiment tracking / reporting
"""
import wandb

def instantiate(which="wandb"):
    """Instantiate experiment tracker

    Args:
        which (str, optional): Tracker name. Defaults to "wandb".
    """
    if which == "wandb":
        return WandbTracker()
    if which == "none" or which is None:
        return Tracker()
    raise NotImplementedError(f"Undefined tracker {which}")

class Tracker:
    """Abstraction for experiment tracking / reporting
    """
    def log(self, key_value: dict):
        pass
    def update_config(self, config: dict):
        pass

class WandbTracker:
    """Weights and biases
    """
    def __init__(self):
        wandb.init(project="decal", entity="sog")

    def update_config(self, config: dict):
        wandb.config.update(config)

    def log(self, key_value: dict):
        wandb.log(key_value)