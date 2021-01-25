# -*- coding: utf-8 -*-
"""
Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import pulp
import numpy as np

def train(env, **kwargs):
    # do nothing; included for compatibility with RL agent
    return MPC_agent(env)


class MPC_agent:

    def __init__(self, env):

        self.env = env

        self.forecast_length = env.forecast_length
        self.indices = [str(i) for i in range(self.forecast_length)]

        self.gen1_vars = pulp.LpVariable.dicts(name="gen1", indexs=self.indices, 
                                                lowBound=env.param.generator_1_min, upBound=env.param.generator_1_max)
        self.gen2_vars = pulp.LpVariable.dicts(name="gen2", indexs=self.indices, 
                                                lowBound=env.param.generator_2_min, upBound=env.param.generator_2_max)
        self.imb_vars = pulp.LpVariable.dicts(name="imb", indexs=self.indices)
        self.imb_plus_ind = pulp.LpVariable.dicts(name="imb_plus_ind", indexs=self.indices, lowBound=0, upBound=1, cat=pulp.LpBinary)
        self.imb_plus_vars = pulp.LpVariable.dicts(name="imb_plus", indexs=self.indices, lowBound=0)
        self.imb_minus_vars = pulp.LpVariable.dicts(name="imb_minus", indexs=self.indices, lowBound=0)

        self.problem = pulp.LpProblem(name="dispatch-problem", sense=pulp.constants.LpMinimize)

        # add ramp constraints
        for i in range(len(self.indices) - 1):
            self.problem += self.gen1_vars[self.indices[i+1]] - self.gen1_vars[self.indices[i]] <= self.env.param.ramp_1_max
            self.problem += self.gen1_vars[self.indices[i+1]] - self.gen1_vars[self.indices[i]] >= self.env.param.ramp_1_min
            self.problem += self.gen2_vars[self.indices[i+1]] - self.gen2_vars[self.indices[i]] <= self.env.param.ramp_2_max
            self.problem += self.gen2_vars[self.indices[i+1]] - self.gen2_vars[self.indices[i]] >= self.env.param.ramp_2_min

        # define constraints for non-linear costs
        M = 100
        for i, idx in enumerate(self.indices):
            # define imb_plus_ind as the sign of imb_vars
            self.problem += self.imb_vars[idx] <= M * self.imb_plus_ind[idx]
            self.problem += self.imb_vars[idx] >= -M * (1 - self.imb_plus_ind[idx])
            # define imb_plus_vars as the positive part of imb_vars
            self.problem += self.imb_plus_vars[idx] >= self.imb_vars[idx]
            self.problem += self.imb_plus_vars[idx] <= self.imb_vars[idx] + M*(1 - self.imb_plus_ind[idx])
            self.problem += self.imb_plus_vars[idx] >= -M * self.imb_plus_ind[idx]
            self.problem += self.imb_plus_vars[idx] <= M * self.imb_plus_ind[idx]
            # define imb_mins_vars as the negative part of imb_vars
            self.problem += self.imb_minus_vars[idx] == self.imb_plus_vars[idx] - self.imb_vars[idx]

        return

    def _add_current_constraints(self, current_gen, forecast, steps_in_objective=None):

        # if steps_in_objective is given, use only a subset of terms in the objective function
        if steps_in_objective is None:
            objective_indices = self.indices
        else:
            assert steps_in_objective <= len(forecast), f"steps_in_objective={steps_in_objective} should be <= {len(forecast)}"
            objective_indices = [self.indices[i] for i in range(steps_in_objective)]

        # define objective function
        self.problem.objective = self.env.param.generator_1_cost * pulp.lpSum([self.gen1_vars[i] for i in objective_indices]) \
            + self.env.param.generator_2_cost * pulp.lpSum([self.gen2_vars[i] for i in objective_indices]) \
            + self.env.param.imbalance_cost_factor_high * pulp.lpSum([self.imb_minus_vars[i] for i in objective_indices]) \
            + self.env.param.imbalance_cost_factor_low * pulp.lpSum([self.imb_plus_vars[i] for i in objective_indices])

        # in this function we use named constraints, which are a bit more fiddly, but they are replaced when the function is called again

        # add ramp rates for current time step
        self.problem.constraints['gen1up'] = pulp.LpConstraint(self.gen1_vars[self.indices[0]] - current_gen[0], rhs=self.env.param.ramp_1_max, sense=pulp.LpConstraintLE)
        self.problem.constraints['gen1down'] = pulp.LpConstraint(self.gen1_vars[self.indices[0]] - current_gen[0], rhs=self.env.param.ramp_1_min, sense=pulp.LpConstraintGE)
        self.problem.constraints['gen2up'] = pulp.LpConstraint(self.gen2_vars[self.indices[0]] - current_gen[1], rhs=self.env.param.ramp_2_max, sense=pulp.LpConstraintLE)
        self.problem.constraints['gen2down'] = pulp.LpConstraint(self.gen2_vars[self.indices[0]] - current_gen[1], rhs=self.env.param.ramp_2_min, sense=pulp.LpConstraintGE)
        
        for i, idx in enumerate(self.indices):
            # identify the actual imbalance, given the inputs
            self.problem.constraints['imb_'+idx] = pulp.LpConstraint(self.gen1_vars[idx] + self.gen2_vars[idx] - forecast[i] - self.imb_vars[idx], rhs=0, sense=pulp.LpConstraintEQ)

        return

    def predict(self, obs, deterministic=True, **kwargs):
        
        # get the current time (minus one) and map it to an integer
        current_time = int(round(obs[0]))   
        actual_forecast_length = min(self.env.forecast_length, self.env.param.steps_per_episode - current_time - 1)

        # set constraints on the basis of current output and forecast
        self._add_current_constraints(current_gen=(obs[1], obs[2]), forecast=obs[3:], steps_in_objective=actual_forecast_length)

        # solve the MILP and suppress output
        self.problem.solve(pulp.PULP_CBC_CMD(msg=False))

        # extract t=0 actions for generators
        a1 = self.gen1_vars[self.indices[0]].varValue
        a2 = self.gen2_vars[self.indices[0]].varValue

        return (a1, a2), None

    def full_solution(self, obs):

        # set constraints on the basis of current output and forecast
        self._add_current_constraints(current_gen=(obs[1], obs[2]), forecast=obs[3:], steps_in_objective=self.env.forecast_length)

        # solve the MILP and suppress output
        self.problem.solve(pulp.PULP_CBC_CMD(msg=False))

        return pulp.value(self.problem.objective)