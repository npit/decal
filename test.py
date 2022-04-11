from stable_baselines3.common.env_checker import check_env
from deceptive_alignment.deceptive_alignment import DeceptiveAlignment

env = DeceptiveAlignment()
check_env(env)