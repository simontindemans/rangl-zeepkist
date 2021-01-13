import gym
import reference_environment

import logging
import time

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PlotWrapper(gym.Wrapper):
    """
    Wrapper that modifies the plot function to show realised instead of forecast values in the bottom right.
    """

    def __init__(self, env):
        super().__init__(env)
        
    def plot(self, fname):
        state = self.state

        fig, ax = plt.subplots(2, 2)

        # cumulative total cost
        plt.subplot(221)
        plt.plot(np.cumsum(state.rewards_all))
        plt.xlabel("time")
        plt.ylabel("cumulative reward")
        plt.tight_layout()
        # could be expanded to include individual components of the reward

        # generator levels
        plt.subplot(222)
        plt.plot(np.array(state.generator_1_levels_all))
        plt.plot(np.array(state.generator_2_levels_all))
        plt.xlabel("time")
        plt.ylabel("generator levels")
        plt.tight_layout()


        # actions
        plt.subplot(223)
        plt.plot(np.array(state.actions_all))
        plt.xlabel("time")
        plt.ylabel("actions")
        plt.tight_layout()


        # agent predictions
        plt.subplot(224)
        plt.plot(np.diagonal(np.array(state.agent_predictions_all)))
        plt.xlabel("time")
        plt.ylabel("realistations")
        plt.tight_layout()


        plt.savefig(fname)


class ResetWrapper(gym.Wrapper):
    """
    Wrapper that modifies the reset() function to randomise the second peak.
    """
    
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        # dirty override of a private parameter
        # this creates an instance variable instead of the class variable in Parameters()
        self.env.param.second_peak_time = np.random.randint(low=10, high=95)
        return super().reset()

class ObsWrapper(gym.ObservationWrapper):
    """
    Wrapper for observations.

    Modifies the observation vector so that the forecast part (values 2:98) consists of the current value and 
    immediate forecast, padded with repeats of the final element. This should make the policy time-invariant.
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        obs_header = observation[0:3]
        # include the 'current' value for t=-1 for consistency
        extended_demand_series = (2,) + observation[3:]
        # take only the part from t until the future
        obs_forecast = extended_demand_series[obs_header[0] + 1:]
        # pad the forecast with copies of the final value
        padding_required = self.env.param.steps_per_episode + 1 - len(obs_forecast)
        padded_forecast = np.pad(obs_forecast, (0,padding_required), 'edge')
        # return an observation vector of the same length
        return obs_header + tuple(padded_forecast)[:self.env.param.steps_per_episode]


class ActWrapper(gym.ActionWrapper):
    """
    Wrapper for actions.

    Adjust the actions to lie in a [-1,1] 2D box, adapted to the ramp rates of the generators.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1,1,(2,))

    def action(self, action):
        offset_1 = action[0]*self.env.param.ramp_1_max if action[0] >= 0 else - action[0]*self.env.param.ramp_1_min
        offset_2 = action[1]*self.env.param.ramp_2_max if action[1] >= 0 else - action[1]*self.env.param.ramp_2_min
        a1 = self.env.state.generator_1_level + offset_1
        a2 = self.env.state.generator_2_level + offset_2
        return (a1, a2)



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
    env.plot(plot_name)
    return

# create the environment, including action/observation adaptations defined above
base_env = gym.make("reference_environment:reference-environment-v0")
env = ResetWrapper(PlotWrapper(ActWrapper(ObsWrapper(base_env))))

# Train an RL agent on the environment
model = train_rl(env, models_to_train=1, episodes_per_model=500)

# Perform two independent runs
env.seed(42)
run_rl(model, env, "agent_run_42.png")

env.seed(1234)
run_rl(model, env, "agent_run_1234.png")


