import gym
import matplotlib.pyplot as plt
import numpy as np

from rl2020.exercise3.train_dqn import CARTPOLE_CONFIG as DQN_CARTPOLE_CONFIG
from rl2020.exercise3.train_dqn import LUNARLANDER_CONFIG as DQN_LUNARLANDER_CONFIG
from rl2020.exercise3.train_dqn import train as dqn_train
from rl2020.exercise3.train_reinforce import (
    CARTPOLE_CONFIG as REINFORCE_CARTPOLE_CONFIG,
)
from rl2020.exercise3.train_reinforce import (
    LUNARLANDER_CONFIG as REINFORCE_LUNARLANDER_CONFIG,
)
from rl2020.exercise3.train_reinforce import train as reinforce_train


plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"font.size": 15})

TRAINING_RUNS = 5
EVAL_FREQ = 1000
EVAL_EPISODES = 10

CARTPOLE_TIMESTEPS = 100000
CARTPOLE_CONFIGS = [
    (DQN_CARTPOLE_CONFIG, "DQN", dqn_train),
    (REINFORCE_CARTPOLE_CONFIG, "REINFORCE", reinforce_train),
]

LUNARLANDER_TIMESTEPS = 500000
LUNARLANDER_CONFIGS = [
    (DQN_LUNARLANDER_CONFIG, "DQN", dqn_train),
    (REINFORCE_LUNARLANDER_CONFIG, "REINFORCE", reinforce_train),
]

CONFIGS = CARTPOLE_CONFIGS
# CONFIGS = LUNARLANDER_CONFIGS

def prepare_config(config, alg_name, train_f):
    """
    Add further parameters to configuration file used in evaluation for plots

    :param config (Dict): configuration file to extend
    :param alg_name (str): name of algorithm for this configuration
    :param train_f (Callable): training function of algorithm
    """
    env_name = config['env'][:-3]
    if env_name.lower() == "cartpole":
        max_timesteps = CARTPOLE_TIMESTEPS
    elif env_name.lower() == "lunarlander":
        max_timesteps = LUNARLANDER_TIMESTEPS

    config["alg"] = alg_name
    config["train"] = train_f
    config["max_timesteps"] = max_timesteps
    config["eval_freq"] = EVAL_FREQ
    config["eval_episodes"] = EVAL_EPISODES


def plot_timesteps(values: np.ndarray, xlabel: str, ylabel: str, legend_name: str):
    """
    Plot values with respect to timesteps
    
    :param values (np.ndarray): numpy array of values to plot as y values
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    x_values = EVAL_FREQ + np.arange(len(values)) * EVAL_FREQ
    plt.plot(x_values, values, label=f"{legend_name}")
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.3)


if __name__ == "__main__":
    # execute training and evaluation to generate return plots
    plt.figure(figsize=(8, 8))
    axes = plt.gca()
    axes.set_ylim([0,200])
    hlines = False

    env_name = None
    for config, name, train in CONFIGS:
        env_name = config['env'][:-3]
        prepare_config(config, name, train)

        plt.title(f"Average Returns on {env_name}")

        # draw threshold line
        if hlines == False:
            x_min = 0
            x_max = config["max_timesteps"]
            if env_name.lower() == "lunarlander":
                plt.hlines(y=190, xmin=x_min, xmax=x_max, colors='k', linestyles='dotted', label="LunarLander threshold")
            elif env_name.lower() == "cartpole":
                plt.hlines(y=195, xmin=x_min, xmax=x_max, colors='k', linestyles='dotted', label="Cartpole threshold")
            hlines = True

        print(f"{config['alg']} performance on {env_name}")

        env = gym.make(config["env"])

        num_returns = int(config["max_timesteps"] / config["eval_freq"])

        eval_returns = np.zeros((TRAINING_RUNS, num_returns))
        for i in range(TRAINING_RUNS):
            print(f"Executing training for {name} - run {i + 1}")
            returns, _ = config["train"](env, config, output=False)
            # correct for missing returns (repeat last one)
            if returns.shape[-1] < eval_returns.shape[-1]:
                returns_extended = np.zeros(num_returns)
                returns_extended[: returns.shape[-1]] = returns
                returns_extended[returns.shape[-1] :] = returns[-1]
                returns = returns_extended
            eval_returns[i, :] = returns
        returns_total = eval_returns.mean(axis=0)
        plot_timesteps(returns_total, "Timestep", "Mean Eval Returns", name)

    assert env_name is not None
    plt.savefig(f"{env_name.lower()}_results.pdf", format="pdf")
    plt.show()
