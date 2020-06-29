from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2020.exercise3.networks import FCNetwork
from rl2020.exercise3.replay import Transition, ReplayBuffer


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space

        :attr saveables (Dict[str, torch.nn.Module]):
            mapping from network names to PyTorch network modules
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str) -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "{path}"

        :param path (str): path to directory where to save models
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        """Returns an action to select in given observation

        **DO NOT CHANGE THIS FUNCTION**
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **DO NOT CHANGE THIS FUNCTION**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        """Updates model parameters

        **DO NOT CHANGE THIS FUNCTION**
        """
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU MUST COMPLETE THIS CLASS**
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma

        :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
        :attr critics_target (FCNetwork): fully connected DQN target network
        :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
        :attr update_counter (int): counter of updates for target network updates
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
        )
        self.loss = torch.nn.MSELoss()

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        # ######################################### #
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )
        self.update_mode = "hard"
        self.soft_update_tua = 0.5
        self.epsilon = 0.9

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        max_deduct, decay = 0.95, 0.07
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct
        # self.update_mode = "hard"
        self.update_mode = "soft"
        self.soft_update_tua = 0.6
        # self.gamma = 1.0
        return

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore (or act greedily)
        :return (sample from self.action_space): action the agent should perform
        """
        act_vals = self.critics_net.forward(Tensor(obs))
        max_val = torch.max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
        if explore:
            if np.random.uniform() < self.epsilon:
                return np.random.randint(0, self.action_space.n)
            else:
                return np.random.choice(max_acts)
        else:
            return np.random.choice(max_acts)

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        
        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network and return the Q-loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to value of losses
        """
        # Update target network
        if self.update_counter % self.target_update_freq == 0:
            if self.update_mode == "hard":
                self.critics_target.hard_update(self.critics_net)
            elif self.update_mode == "soft":
                self.critics_target.soft_update(self.critics_net, self.soft_update_tua)
        self.update_counter += 1

        # Indicate index
        batch_states = batch[0]
        batch_actions = batch[1]
        batch_next_states = batch[2]
        batch_rewards = batch[3]
        batch_done = batch[4]

        # Predict and calculate loss
        q_eval = self.critics_net.forward(batch_states).gather(1, batch_actions.long())
        q_next = self.critics_target.forward(batch_next_states).detach()
        q_target = batch_rewards + self.gamma * (1 - batch_done) * torch.max(q_next, dim=1)[0].view(self.batch_size, 1)
        q_loss = self.loss(q_eval, q_target)

        self.critics_optim.zero_grad()
        q_loss.backward()
        self.critics_optim.step()

        return {"q_loss": q_loss}


class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3

    **YOU MUST COMPLETE THIS CLASS**
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma

        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for policy network
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update({"policy": self.policy})

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        max_deduct, decay = 0.95, 0.07
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timesteps))) * max_deduct
        return

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        probs = self.policy.forward(Tensor(obs)).detach()
        dist = torch.nn.functional.softmax(probs, dim=-1)
        m = Categorical(dist)
        action = m.sample()
        return action.item()

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int]
    ) -> Dict[str, float]:
        """Update function for REINFORCE

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        G = self.compute_gt(rewards)
        p_loss = 0.0
        for i in range(len(rewards)):
            probs = self.policy.forward(Tensor(observations[i]))
            dist = torch.nn.functional.softmax(probs, dim=-1)
            m = Categorical(dist)
            p_loss -= m.log_prob(torch.FloatTensor([actions[i]])) * G[i]

        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return {"p_loss": p_loss}

    def compute_gt(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.gamma
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))
