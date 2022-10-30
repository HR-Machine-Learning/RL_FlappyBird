from multiprocessing.dummy import active_children
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple
from sklearn.neural_network import MLPRegressor
from abst_flappy_agent import FlappyAgent


class NewDeepQAgent(FlappyAgent):
    def __init__(self):
        self.learning_rate: float = 0.1  # alpha
        self.discount: float = 1  # gamma
        self.epsilon: float = 0.1
        self.fig = None
        self.nbr_of_episodes = 0
        self.nbr_of_frames = 0
        self.initialize_network()

    def initialize_network(self):
        ''' To the neural network we feed in the state, pass that 
        through several hidden layers and then output the Q-values.
        This means 4 input nodes, 1 output node'''
        self.model: MLPRegressor = MLPRegressor(hidden_layer_sizes=(100, 10),
                                                activation='logistic',
                                                learning_rate_init=0.1)
        # the ndarray is automatically filled with values
        X = np.ndarray(shape=(4, 4), dtype=int)  # state
        y = np.ndarray(shape=(4), dtype=float)  # expected Q-values
        self.model.fit(X, y)

    def get_q(self, state: Dict[str, int]) -> float:
        # Returns Q(s) based on a state
        internal_state: np.array = self.state_to_internal_state(state)

        return self.model.predict(internal_state)

    def set_q(self, state: Dict[str, int], action: int, o_value: float) -> None:
        # Sets a value to Q(s) based on a state and action
        y = np.array(o_value)
        self.model.partial_fit(self.state_to_internal_state(state), y)
        # TODO this will most likely lead to an error, check needed

    def observe(self, iteration, s1: Dict[str, int], action: int, reward: float, s2: Dict[str, int], end: bool) -> None:
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        current_state_value: float = self.get_q(s1)

        if current_state_value is None:
            current_state_value = 0

        # TODO Bellman's equation needs to be updated for function approximation version
        o_value: float = current_state_value + self.learning_rate * \
            (reward + self.discount * self.get_q(s2,
             self.action_with_max_value(s2)) - self.get_q(s1, action))

        self.set_q(s1, o_value)

        if end:
            self.nbr_of_episodes += 1

    def action_with_max_value(self, state: Dict[str, int]) -> int:
        # TODO needs to be deleted
        s_0 = self.get_q(state, 0)
        s_1 = self.get_q(state, 1)

        if s_0 is None:
            s_0 = 0
        if s_1 is None:
            s_1 = 0
        if s_1 > s_0:
            return 1
        else:
            return 0

    def training_policy(self, state: Dict[str, int]) -> int:
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        greedy: bool = np.random.choice(
            [False, True], p=[self.epsilon, 1 - self.epsilon])
        # greedy = False

        if greedy:
            action = self.action_with_max_value(state)
        else:
            action = random.randint(0, 1)
        # action = [0, 0, 1, 1, 1, 1, 1][random.randint(0, 6)]

        return action

    def policy(self, state: Dict[str, int]) -> int:
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        return self.action_with_max_value(state)

    def state_to_internal_state(self, state: Dict[str, int]) -> np.array:
        # Normalizes the state in the range [-1,1]
        player_y_normalized = 2 * (state['player_y'] - 0) / (512 - 0) - 1
        next_pipe_top_y_normalized = 2 * \
            (state['next_pipe_top_y'] - 0) / (512 - 0) - 1
        next_pipe_dist_to_player_normalized = 2 * \
            (state['next_pipe_dist_to_player'] - 0) / (288 - 0) - 1
        player_vel_normalized = 2 * (state['player_vel'] - -8) / (10 - -8) - 1

        normalized_state = np.array([player_y_normalized],
                                    [next_pipe_top_y_normalized],
                                    [next_pipe_dist_to_player_normalized],
                                    [player_vel_normalized])

        # TODO remove this print
        print(normalized_state)

        return normalized_state

    def translate_int(self, value: int, leftMin: int, leftMax: int, rightMin: int, rightMax: int) -> int:
        # Figure out how 'wide' each range is
        leftSpan: int = leftMax - leftMin
        rightSpan: int = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled: float = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return max(min(floor(rightMin + (valueScaled * rightSpan)), rightMax), rightMin)
