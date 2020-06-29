import matplotlib.pyplot as plt
import numpy as np
from typing import List


def act_to_str(act: int):
    """
    Map FrozenLake action index to human-readable action name
    
    :param act (int): action index of FrozenLake
    :return (str): human-readable action name
    """
    if act == 0:
        return "L"
    elif act == 1:
        return "D"
    elif act == 2:
        return "R"
    elif act == 3:
        return "U"
    else:
        raise ValueError("Invalid action value")


def wolf_visualize_policy(policy: List[float], player: int):
    """
    Plot visualization of Wolf-PHC policy

    :param policy (List[float]): player policy as probability distribution for RPS game
    :param player (int): player id (EITHER 1 or 2!)
    """
    src_point = np.asarray(
        [[policy[idx][0], policy[idx][1]] for idx in range(len(policy) - 1)]
    )
    dst_point = np.asarray(
        [[policy[idx + 1][0], policy[idx + 1][1]] for idx in range(len(policy) - 1)]
    )

    for src, dst in zip(src_point, dst_point):
        plt.plot([src[0]], [src[1]], marker="o", markersize=3, color="red")
        plt.plot([src[0], dst[0]], [src[1], dst[1]], "k-")

    plt.plot(
        [dst_point[-1][0]], [dst_point[-1][1]], marker="o", markersize=3, color="red"
    )

    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xlabel("Pr(Rock)")
    plt.ylabel("Pr(Paper)")
    if player == 1:
        plt.title("RPS 1st player policy visualisation")
    elif player == 2:
        plt.title("RPS 2nd player policy visualisation")
    plt.savefig(f"q2_wolf_agent{player}_pi.pdf", format="pdf")

    plt.show()


def visualise_q_table(q_table):
    """
    Print visualisation of Q-table for FrozenLake environment

    :param q_table (Dict[(Obs, Act), float]): Q-table to visualise as prints
    """
    # extract best acts
    act_table = np.zeros((4, 4))
    str_table = []
    for row in range(4):
        str_table.append("")
        for col in range(4):
            pos = row * 4 + col
            max_q = None
            max_a = None
            for a in range(4):
                q = q_table[(pos, a)]
                if max_q is None or q > max_q:
                    max_q = q
                    max_a = a
            act_table[row, col] = max_a
            str_table[row] += act_to_str(max_a)

    # print best actions in human_readable format
    print("\nAction selection table:")
    for row_str in str_table:
        print(row_str)
    print()


def evaluate(env, agent, eval_episodes, render, output=True):
    """
    Evaluate configuration on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param agent (Agent): agent to act in environment
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation results should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    episodic_returns = []
    for eps_num in range(eval_episodes):
        obs = env.reset()
        if render:
            env.render()
        episodic_return = 0
        done = False

        while not done:
            act = agent.act(obs)
            n_obs, reward, done, info = env.step(act)
            if render:
                env.render()

            episodic_return += reward

            obs = n_obs

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns)
    std_return = np.std(episodic_returns)

    if output:
        print(f"EVALUATION: MEAN RETURN OF {mean_return}")
    return mean_return, std_return
