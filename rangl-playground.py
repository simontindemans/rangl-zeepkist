import logging
import time

import numpy as np
import gym
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

import reference_environment
import provided.util as util
import envwrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def train_rl(env, models_to_train=40, episodes_per_model=100):
    """
    RL training function. 
    """

    # using SAC - adjusted gamma to a lower value due to the relatively fast response of the system
    model = SAC(MlpPolicy, env, verbose=1, gamma=0.85)
    start = time.time()

    for i in range(models_to_train):
        steps_per_model = episodes_per_model * env.param.steps_per_episode
        model.learn(total_timesteps=steps_per_model)
        model.save("MODEL_" + str(i))

    end = time.time()
    print("time (min): ", (end - start) / 60)

    # return final model
    return model

def run_rl(model, env, plot_name):
    observation = env.reset()
    done = False
    while not done:
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        # note: using 'deterministic = True' for evaluation fixes the SAC policy
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    # plot the episode using the modified function (includes realisations at the bottom right)
    if plot_name is not None:
        env.plot(plot_name)
    return np.sum(env.state.rewards_all)

# create the environment, including action/observation adaptations defined in the envwrapper module
base_env = gym.make("reference_environment:reference-environment-v0")
env = envwrapper.PlotWrapper(envwrapper.ActWrapper(envwrapper.EfficientObsWrapper(base_env, obs_length=25)))

# Train an RL agent on the environment
agent = train_rl(env, models_to_train=1, episodes_per_model=500)

# Perform two independent runs
run_rl(agent, env, "agent_run_1.png")
run_rl(agent, env, "agent_run_2.png")

# collect results over 50 independent runs, display summary statistics
result_list = np.zeros(50)
for i in range(len(result_list)):
    result_list[i] = run_rl(agent, env, None)

print(f"Summary of 50 results:")
print(f"Mean: {np.mean(result_list)}")
print(f"Std: {np.std(result_list)}")
print(f"Min: {np.min(result_list)}")
print(f"Max: {np.max(result_list)}")

evaluate = util.Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="provided/seeds.csv")
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)

print('Mean reward:',mean_reward)


