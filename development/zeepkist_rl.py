import time

import numpy as np
import gym
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy


def train(env, models_to_train=1, episodes_per_model=100, **kwargs):
    """
    RL training function. 
    """

    # using SAC - adjusted gamma to a lower value due to the relatively fast response of the system
    model = SAC(MlpPolicy, env, **kwargs)
    start = time.time()

    for i in range(models_to_train):
        steps_per_model = episodes_per_model * env.param.steps_per_episode
        model.learn(total_timesteps=steps_per_model)
        model.save("output/MODEL_" + str(i))

    end = time.time()
    print("time (min): ", (end - start) / 60)

    # return final model
    return model