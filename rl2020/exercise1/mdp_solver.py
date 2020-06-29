from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2020.utils import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        
        :attr action_dim (int): number of actions in the MDP
        :attr state_dim (int): number of states in the MDP
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP

        **DO NOT CHANGE THIS FUNCTION**
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm

    **YOU MUST COMPLETE THIS CLASS**
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

        Useful Variables:
        1. `self.mdp` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        V = np.zeros(self.state_dim)
        while True:
            delta = 0
            for i, state in enumerate(self.mdp.states):
                A = self._calc_action(state, i, V)
                best_action_V = np.max(A)
                delta = max(delta, np.abs(best_action_V - V[i]))
                V[i] = best_action_V
            if delta < theta:
                break
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        for i, state in enumerate(self.mdp.states):
            A = self._calc_action(state, i, V)
            best_action = np.argmax(A)
            policy[i, best_action] = 1.0
        return policy

    def _calc_action(self, state, _state, V):
        A = np.zeros(self.action_dim)
        for j, action in enumerate(self.mdp.actions):
            for transition in [t for t in self.mdp.transitions if t.state == state and t.action == action]:
                next_state = transition.next_state
                _next_state = self.mdp._state_dict[next_state]
                prob = self.mdp.P[_state, j, _next_state]
                reward = self.mdp.R[_state, j, _next_state]
                A[j] += prob * (reward + self.gamma * V[_next_state])
        return A


    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm

    **YOU MUST COMPLETE THIS CLASS**
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)
        while True:
            delta = 0
            for i, state in enumerate(self.mdp.states):
                v = 0
                for j, action in enumerate(self.mdp.actions):
                    for transition in [t for t in self.mdp.transitions if t.state == state and t.action == action]:
                        action_prob = policy[i, j]
                        next_state = transition.next_state
                        _next_state = self.mdp._state_dict[next_state]
                        prob = self.mdp.P[i, j, _next_state]
                        reward = self.mdp.R[i, j, _next_state]
                        v += action_prob * prob * (reward + self.gamma * V[_next_state])
                delta = max(delta, np.abs(v - V[i]))
                V[i] = v
            if delta < self.theta:
                break
        return V

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes one policy improvement iteration

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**
        
        Useful Variables (As with Value Iteration):
        1. `self.mdp` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        # Start with a (random) policy
        policy = np.zeros([self.state_dim, self.action_dim])
        V = np.zeros([self.state_dim])
        while True:
            V = self._policy_eval(policy)
            is_policy_stable = True
            for i, state in enumerate(self.mdp.states):
                best_action_from_policy = np.argmax(policy[i])
                A = self._calc_action(state, i, V)
                best_action_true = np.argmax(A)
                if best_action_from_policy != best_action_true:
                    is_policy_stable = False
                policy[i] = np.eye(len(self.mdp.actions))[best_action_true]

            if is_policy_stable:
                return policy, V

    def _calc_action(self, state, _state, V):
        A = np.zeros(self.action_dim)
        for j, action in enumerate(self.mdp.actions):
            for transition in [t for t in self.mdp.transitions if t.state == state and t.action == action]:
                next_state = transition.next_state
                _next_state = self.mdp._state_dict[next_state]
                prob = self.mdp.P[_state, j, _next_state]
                reward = self.mdp.R[_state, j, _next_state]
                A[j] += prob * (reward + self.gamma * V[_next_state])
        return A

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    # DISCLAIMER
    # To create valid MDPs, a transition for each action should be defined for each state.
    # Our framework only supports MDPs where each action is considered valid in each state.
    # Hence, we will only test your implementation on such, in our tool valid, MDPs.
    mdp = MDP()
    mdp.add_transition(
        #           start   action  end  prob  reward
        Transition("high", "wait", "high", 1, 2),
        Transition("high", "search", "high", 0.8, 5),
        Transition("high", "search", "low", 0.2, 5),
        Transition("high", "recharge", "high", 1, 0),
        Transition("low", "recharge", "high", 1, 0),
        Transition("low", "wait", "low", 1, 2),
        Transition("low", "search", "high", 0.6, -3),
        Transition("low", "search", "low", 0.4, 5),
    )

    solver = ValueIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)
