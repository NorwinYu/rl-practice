from .mdp import MDP, Transition, State, Action
from gym.envs.registration import register

register(id="rps-v0", entry_point="rl2020.utils.rps:RPS")

register(
    id="FrozenLakeNotSlippery-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)
