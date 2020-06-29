import gym

from rl2020.exercise2.agents import QLearningAgent
from rl2020.exercise2.utils import visualise_q_table, evaluate


CONFIG = {
    "total_eps": 50000,
    "eps_max_steps": 100,
    "eval_freq": 5000,
    "gamma": 0.99,
    "alpha": 0.05,
    "epsilon": 0.9,
}


def q_learning_eval(env, config, q_table, eval_episodes=100, render=False, output=True):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    return evaluate(env, eval_agent, eval_episodes, render, output)


def train(env, config, output=True):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_return_stds = []

    for eps_num in range(config["total_eps"]):
        obs = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)
            n_obs, reward, done, _ = env.step(act)
            agent.learn(obs, act, reward, n_obs, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, std_return = q_learning_eval(
                env, config, agent.q_table, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)

    return total_reward, evaluation_return_means, evaluation_return_stds, agent.q_table


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = QLearningAgent(
        action_space=env.action_space, obs_space=env.observation_space, **CONFIG
    )

    total_reward, _, _, q_table = train(env, CONFIG)
    print()
    print(f"Total reward over training: {total_reward}\n")
    print("Q-table:")
    print(q_table)
    visualise_q_table(q_table)
