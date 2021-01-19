import logging

import numpy as np
import gym

# rangl provided modules
import reference_environment
import provided.util as util

# own modules
import envwrapper
import zeepkist_rl
import zeepkist_mpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_episode(env, agent, plot_name = None):
    observation = env.reset()
    done = False
    while not done:
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        # note: using 'deterministic = True' for evaluation fixes the SAC policy
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    # plot the episode using the modified function (includes realisations at the bottom right)
    if plot_name is not None:
        env.plot(plot_name)
    return np.sum(env.state.rewards_all)

# create the environment, including action/observation adaptations defined in the envwrapper module
base_env = gym.make("reference_environment:reference-environment-v0")
env = envwrapper.PlotWrapper(envwrapper.ActWrapper(envwrapper.EfficientObsWrapper(base_env, obs_length=25)))
env = envwrapper.PlotWrapper(envwrapper.EfficientObsWrapper(base_env, obs_length=25))

# Train an RL agent on the environment
agent = zeepkist_mpc.train(env, episodes_per_model=500, verbose=1, gamma=0.85)

# Perform two independent runs
run_episode(env, agent, "agent_run_1.png")
run_episode(env, agent, "agent_run_2.png")

# collect results over 50 independent runs, display summary statistics
result_list = np.zeros(50)
for i in range(len(result_list)):
    result_list[i] = run_episode(env, agent)

print(f"Summary of 50 results:")
print(f"Mean: {np.mean(result_list)}")
print(f"Std: {np.std(result_list)}")
print(f"Min: {np.min(result_list)}")
print(f"Max: {np.max(result_list)}")

evaluate = util.Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="provided/seeds.csv")
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)

print('Mean reward:',mean_reward)


