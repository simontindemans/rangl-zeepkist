# -*- coding: utf-8 -*-
"""
Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

# ADAPTED FROM env.py - copy of model parameters

class Parameters:
    # (Avoid sampling random variables here: they would not be resampled upon reset())
    # problem-specific parameters
    imbalance_cost_factor_high = 50
    imbalance_cost_factor_low = 7
    ramp_1_max = 0.2
    ramp_2_max = 0.5
    ramp_1_min = -0.2
    ramp_2_min = -0.5
    generator_1_max = 3
    generator_1_min = 0.5
    generator_2_max = 2
    generator_2_min = 0.5
    generator_1_cost = 1
    generator_2_cost = 5

    # time parameters
    steps_per_episode = 96
    first_peak_time = 5


class SkeletonEnvironment:

    param = Parameters()  # parameters singleton

    def __init__(self, forecast_length=None):
        if forecast_length is None:
            self.forecast_length = self.param.steps_per_episode
        else:
            self.forecast_length = forecast_length
        return



