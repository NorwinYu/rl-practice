"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import numpy as np
from rl2020.utils import MDP, Transition
from rl2020.exercise1 import ValueIteration, PolicyIteration
from rl2020.exercise2 import QLearningAgent, MonteCarloAgent, WolfPHCAgent

from gym.spaces import Discrete


def test_qagent_0():
    agent = QLearningAgent(
        action_space=Discrete(3),
        obs_space=Discrete(3),
        gamma=0.99,
        alpha=1.0,
        epsilon=0.9,
    )
    agent.schedule_hyperparameters(0, 10)


    assert hasattr(agent, "epsilon")
    assert hasattr(agent, "alpha")
    assert hasattr(agent, "q_table")
    assert hasattr(agent, "gamma")
    assert type(agent.epsilon) == float
    assert type(agent.alpha) == float
    assert agent.epsilon >= 0.0
    assert agent.epsilon <= 1.0

def test_qagent_1():
    agent = QLearningAgent(
        action_space=Discrete(3),
        obs_space=Discrete(3),
        gamma=0.99,
        alpha=1.0,
        epsilon=0.9,
    )
    space = Discrete(10)
    action = space.sample()
    obs = space.sample()
    reward = 0.0
    obs_n = space.sample()

    agent.learn(obs, action, reward, obs_n, False)

    assert (obs, action) in agent.q_table
    assert type(agent.q_table[(obs, action)]) == float


def test_montecarlo_0():
    agent = MonteCarloAgent(
        action_space=Discrete(3), obs_space=Discrete(3), gamma=0.99, epsilon=0.9,
    )
    agent.schedule_hyperparameters(0, 10)

    assert hasattr(agent, "epsilon")
    assert hasattr(agent, "q_table")
    assert hasattr(agent, "gamma")
    assert type(agent.epsilon) == float
    assert agent.epsilon >= 0.0
    assert agent.epsilon <= 1.0

def test_wolf_agent_0():
    agent = WolfPHCAgent(
        gamma=0.99,
        alpha=0.1,
        num_acts=3,
        win_delta=0.025,
        lose_delta=0.1,
        init_policy=[0.4, 0.3, 0.3]
    )


    assert hasattr(agent, "alpha")
    assert hasattr(agent, "n_acts")
    assert hasattr(agent, "gamma")
    assert hasattr(agent, "q_table")
    assert hasattr(agent, "gamma")
    assert hasattr(agent, "init_policy")
    assert hasattr(agent, "avg_pi_table")
    assert hasattr(agent, "pi_table")
    assert hasattr(agent, "vis_table")
    assert hasattr(agent, "win_delta")
    assert hasattr(agent, "lose_delta")

    assert type(agent.win_delta) == float
    assert type(agent.lose_delta) == float
    assert type(agent.alpha) == float
    assert agent.win_delta <= agent.lose_delta


def test_wolf_phc_1():
    agent = WolfPHCAgent(
        gamma=0.99,
        alpha=0.1,
        num_acts=3,
        win_delta=0.1,
        lose_delta=0.025,
        init_policy=[0.4, 0.3, 0.3]
    )
    obs = ""
    action = 1
    reward = -1.0
    obs_n = ""
    done=True

    agent.learn(obs, action, reward, obs_n, True)

    assert (obs, action) in agent.q_table
    assert obs in agent.vis_table
    assert obs in agent.pi_table
    assert obs in agent.avg_pi_table
    assert type(agent.q_table[(obs, action)]) == float
