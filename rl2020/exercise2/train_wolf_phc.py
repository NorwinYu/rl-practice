import gym

import rl2020.utils  # noqa (registers the rps env)
from rl2020.exercise2.agents import WolfPHCAgent
from rl2020.exercise2.utils import wolf_visualize_policy


CONFIG = {
    "total_eps": 1000000,
    "eval_freq": 50000,
    "gamma": 0.99,
    "num_acts": 3,
    "alpha": 0.1,
    "win_delta": 0.000002,
    "lose_delta": 0.000004,
    "init_policy_0": [0.5, 0.38, 0.12],
    "init_policy_1": [0.5, 0.38, 0.12],
}


def train(env, config):
    """
    Train and evaluate Wolf-PHC Agents on the rock-paper-scissors environment using self-play
    with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        rewards, final Q-table, final state-action counts
    """
    agents = []
    for i in range(2):
        config["init_policy"] = config[f"init_policy_{i}"]
        agents.append(WolfPHCAgent(**config))
    eval_policies1 = []
    eval_policies2 = []
    avg_rewards1 = []
    avg_rewards2 = []
    reward_list1 = []
    reward_list2 = []

    total_reward = [0, 0]
    for step_counter in range(config["total_eps"]):
        obs = env.reset()

        acts = [agent.act(ob) for agent, ob in zip(agents, obs)]
        n_obs, rewards, dones, _ = env.step(acts)

        total_reward[0] += rewards[0]
        total_reward[1] += rewards[1]

        for agent, ob, act, rew, n_ob, done in zip(
            agents, obs, acts, rewards, n_obs, dones
        ):
            agent.learn(ob, act, rew, n_ob, done)

        reward_list1.append(rewards[0])
        reward_list2.append(rewards[1])

        if step_counter % config["eval_freq"] == 0:
            # Store the current policies of agents
            eval_policies1.append(agents[0].pi_table[obs[0]].copy())
            eval_policies2.append(agents[1].pi_table[obs[1]].copy())

            # Average the rewards achieved since previous evaluation
            avg_rewards1.append(sum(reward_list1) / len(reward_list1))
            avg_rewards2.append(sum(reward_list2) / len(reward_list2))
            reward_list1 = []
            reward_list2 = []

    return [eval_policies1, eval_policies2], [avg_rewards1, avg_rewards2], total_reward


if __name__ == "__main__":
    env = gym.make("rps-v0")
    eval_policies, avg_rewards, total_reward = train(env, CONFIG)

    print(f"Total reward over training: {total_reward}\n")
    print(f"Average 1st player rewards: {avg_rewards[0]}")
    print(f"Average 2nd player rewards: {avg_rewards[1]}")

    wolf_visualize_policy(eval_policies[0], player=1)
    wolf_visualize_policy(eval_policies[1], player=2)
