from deceptive_alignment.deceptive_alignment import DeceptiveAlignment


# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)
model = lambda env_: env_.action_space.sample()

env = DeceptiveAlignment()
obs = env.reset()
for i in range(10):
    # action, _states = model.predict(obs)
    action = model(env)
    print("Action:", action)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break


