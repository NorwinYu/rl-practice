import gym
import numpy as np
import time
from tqdm import tqdm

from rl2020.exercise3.agents import DQN
from rl2020.exercise3.replay import ReplayBuffer


RENDER = False

LUNARLANDER_CONFIG = {
    "env": "LunarLander-v2",
    "target_return": 190.0,
    "episode_length": 500,
    "max_timesteps": 200000,
    "max_time": 30 * 60,
    "eval_freq": 20000,
    "eval_episodes": 10,
    "learning_rate": 1e-3,
    "hidden_size": (64, 64),
    "target_update_freq": 1000,
    "batch_size": 64,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "save_filename": "dqn_lunarlander_latest.pt",
}

CARTPOLE_CONFIG = {
    "env": "CartPole-v1",
    "target_return": 195.0,
    "episode_length": 200,
    "max_timesteps": 50000,
    "max_time": 30 * 60,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "learning_rate": 1e-3,
    "hidden_size": (64,),
    "target_update_freq": 1000,
    "batch_size": 64,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "save_filename": None,
}


# CONFIG = CARTPOLE_CONFIG
CONFIG = LUNARLANDER_CONFIG


def play_episode(
    env,
    agent,
    replay_buffer,
    train=True,
    explore=False,
    render=False,
    max_steps=200,
    batch_size=64,
):
    """
    Play out a single episode

    :param env (gym.Env): gym environment to use
    :param agent (DQN): DQN agent to train
    :param replay_buffer (ReplayBuffer): replay buffer to use during training
    :param train (bool): flag whether training should be executed
    :param explore (bool): flag whether agent should use exploration
    :param render (bool): flag whether environment steps should be rendered
    :param max_steps (int): maximum number of steps to take for the episode
    :param batch_size (int): size of update batches to use
    :return (int, float): number of timesteps completed during episode, episode return
    """
    obs = env.reset()
    done = False
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _ = env.step(action)
        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array([action], dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        if max_steps == episode_timesteps:
            break
        obs = nobs

    return episode_timesteps, episode_return


def train(env, config, output=True):
    """     
    Execute training of DQN on given environment using the provided configuration
      
    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = DQN(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            episode_timesteps, _ = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                for _ in range(config["eval_episodes"]):
                    _, episode_return = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                eval_returns_all.append(eval_returns)
                eval_times_all.append(time.time() - start_time)
                if eval_returns >= config["target_return"]:
                    pbar.write(
                        f"Reached return {eval_returns} >= target return of {config['target_return']}"
                    )
                    break

    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_times_all)


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    _ = train(env, CONFIG)
    env.close()
