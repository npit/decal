from deceptive_alignment.deceptive_alignment import DeceptiveAlignment
from policy import get_policy

# parameters
policy_name = "ppo"
policy_name = "random"
# -----------------------

env = DeceptiveAlignment()
model = get_policy(policy_name, env)
model.train()


obs = env.reset()
total_reward = 0
for i in range(10):
    # action, _states = model.predict(obs)
    action = model(env)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    env.render()
    print("Action:", action, "reward:", rewards, "cumulative reward:", total_reward)
    if dones:
        break


