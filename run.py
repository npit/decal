import logging

from deceptive_alignment.deceptive_alignment import DeceptiveAlignment
from reporting import instantiate
from policy import get_policy

# parameters
# -----------------------
policy_name = "random"
policy_name = "ppo"
tracker_name = "wandb"
config = {
    "env": dict(max_train=3, max_deployment=3, modification_reward=-10, mesa_reward=5)
}
# -----------------------
config["num_episodes"] = config.get("num_episodes", 10)
config["max_iterations"] = config.get("max_episodes", 10)
config["train"] = config.get("train", {})


logging.getLogger().setLevel(logging.INFO)
# experiment logger / tracker
tracker = instantiate(tracker_name)
tracker.update_config(config)

# make environment
env = DeceptiveAlignment(**config["env"])

# init and train policy
model = get_policy(policy_name, env, config["train"])

logging.info("Starting training with [{policy_name}]")
model.train()
logging.info("Finished training.")


def evaluate(env, model, num_episodes=10, max_iterations=10):
    episode_rewards = []
    for episode in range(num_episodes):
        logging.info(f"Episode {episode+1} / {num_episodes}")
        obs = env.reset()
        total_episode_reward = 0
        for iteration in range(max_iterations):
            # action, _states = model.predict(obs)
            action, _ = model.predict(obs, env)
            obs, rewards, dones, info = env.step(action)
            total_episode_reward += rewards
            env.render()
            logging.info(f"Action: {action}, reward: {rewards}, episode current reward: {total_episode_reward}")
            if dones:
                break
            tracker.log({"reward": rewards, "iteration": iteration, "episode": episode})
        episode_rewards.append(total_episode_reward)
        logging.info(f"Episode total reward: {total_episode_reward}")
        tracker.log({"total_episode_reward": total_episode_reward})

evaluate(env, model, num_episodes=config["num_episodes"], max_iterations=config["max_iterations"])
