# -*- coding: utf-8 -*-
"""
Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import logging
import csv
import numpy as np
import gym

# rangl provided modules
import reference_environment
import provided.util as util

# own modules
import envwrapper
import plotwrapper
import zeepkist_mpc


class EvaluateClairvoyant:
    """
    Adapted from rangl-provided util.py
    """

    def __init__(self, env, agent):
        self.env = env
        self.param = env.param
        self.agent = agent

    def read_seeds(self, fname="test_set_seeds.csv"):
        file = open(fname)
        csv_file = csv.reader(file)
        seeds = []
        for row in csv_file:
            seeds.append(int(row[0]))
        self.seeds = seeds
        return seeds

    def clairvoyant_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()

            # store the initial generation levels
            initial_action = [self.env.state.generator_1_level, self.env.state.generator_2_level]

            while not self.env.state.is_done():
                # repeat constant action, just in order to get to the end
                self.env.step(initial_action)
            # read realised demand
            realised_demand = np.diagonal(np.array(env.state.agent_predictions_all))
            # optimise the run cost against (clairvoyant) realised demand, pretending to run at t=-1
            min_cost = agent.full_solution([-1] + initial_action + list(realised_demand))
            # collect (negative) cost
            rewards.append(- min_cost)
        return np.mean(rewards)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create the environment, with full-length wrapper
base_env = gym.make("reference_environment:reference-environment-v0")
env = plotwrapper.PlotWrapper(envwrapper.EfficientObsWrapper(base_env, forecast_length=base_env.param.steps_per_episode))

# Initialise MPC agent
agent = zeepkist_mpc.MPC_agent(env)

# evaluate mean performance on competition seeds
evaluate = EvaluateClairvoyant(env, agent)
seeds = evaluate.read_seeds(fname="development/provided/seeds.csv")
mean_reward = evaluate.clairvoyant_agent(seeds)

print('Full phase mean reward:',mean_reward)


