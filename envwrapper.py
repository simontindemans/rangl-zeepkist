import gym

import numpy as np
import matplotlib.pyplot as plt

def obs_transform(observation, env_model):
    obs_header = observation[0:3]
    # take only the part from t until the future
    # repeat the final element so that we do not run out of observations when t=96
    obs_forecast = observation[obs_header[0] + 3 + 1 : ] + (observation[-1],)
    # pad the forecast with copies of the final value (create an array of size 97)
    padding_required = len(observation) -3 + 1 - len(obs_forecast)
    padded_forecast = np.pad(obs_forecast, (0,padding_required), 'edge')
    # return an observation vector of the same length
    return obs_header + tuple(padded_forecast)[:env_model.forecast_length]

def act_transform(action, env_model, current_gen_level):
    offset_1 = action[0]*env_model.param.ramp_1_max if action[0] >= 0 else - action[0]*env_model.param.ramp_1_min
    offset_2 = action[1]*env_model.param.ramp_2_max if action[1] >= 0 else - action[1]*env_model.param.ramp_2_min
    a1 = current_gen_level[0] + offset_1
    a2 = current_gen_level[1] + offset_2
    return (a1, a2)

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
        plt.plot(np.array(state.generator_1_levels_all) + np.array(state.generator_2_levels_all))
        plt.plot(np.array(state.generator_1_levels_all) + np.array(state.generator_2_levels_all) - np.diagonal(np.array(state.agent_predictions_all)))
        plt.xlabel("time")
        plt.ylabel("realised demand and supply, mismatch")
        plt.tight_layout()


        plt.savefig(fname)


class EfficientObsWrapper(gym.ObservationWrapper):
    """
    Wrapper for observations.

    Modifies the observation vector so that the forecast part (values 3:forecast_length+3) consists of the current value and 
    immediate forecast, padded with repeats of the final element if necessary. This should make the policy time-invariant.
    """

    def __init__(self, env, forecast_length=25):
        super().__init__(env)
        assert(1 <= forecast_length <= self.env.param.steps_per_episode,\
            f"Observation length {forecast_length} is outside the permissible range (1, {self.env.param.steps_per_episode})")
        self.forecast_length = forecast_length
        self.observation_space = self._obs_space()

    def observation(self, observation):
        return obs_transform(observation, env_model=self)

    def _obs_space(self):
        # modified from the base environment to restrict the observation length
        obs_low = np.full(self.forecast_length + 3, -1000, dtype=np.float32) # last 'forecast_length' entries of observation are the predictions
        obs_low[0] = -1	# first entry of obervation is the timestep
        obs_low[1] = self.env.param.generator_1_min	# min level of generator 1 
        obs_low[2] = self.env.param.generator_2_min	# min level of generator 2
        obs_high = np.full(self.forecast_length + 3, 1000, dtype=np.float32) # last 96 entries of observation are the predictions
        obs_high[0] = self.env.param.steps_per_episode	# first entry of obervation is the timestep
        obs_low[1] = self.env.param.generator_1_max	# max level of generator 1 
        obs_low[2] = self.env.param.generator_2_max	# max level of generator 2
        result = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        return result


# class ObsWrapper(gym.ObservationWrapper):
#     """
#     Wrapper for observations.

#     Modifies the observation vector so that the forecast part (values 2:98) consists of the current value and 
#     immediate forecast, padded with repeats of the final element. This should make the policy time-invariant.
#     """

#     def __init__(self, env):
#         super().__init__(env)

#     def observation(self, observation):
#         obs_header = observation[0:3]
#         # include the 'current' value for t=-1 for consistency
#         extended_demand_series = (2,) + observation[3:]
#         # take only the part from t until the future
#         obs_forecast = extended_demand_series[obs_header[0] + 1:]
#         # pad the forecast with copies of the final value
#         padding_required = self.env.param.steps_per_episode + 1 - len(obs_forecast)
#         padded_forecast = np.pad(obs_forecast, (0,padding_required), 'edge')
#         # return an observation vector of the same length
#         return obs_header + tuple(padded_forecast)[:self.env.param.steps_per_episode]


class ActWrapper(gym.ActionWrapper):
    """
    Wrapper for actions.

    Adjust the actions to lie in a [-1,1] 2D box, adapted to the ramp rates of the generators.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1,1,(2,))

    def action(self, action):
        return act_transform(action, self.env, (self.env.state.generator_1_level, self.env.state.generator_2_level))
