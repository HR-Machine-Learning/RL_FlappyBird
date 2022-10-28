from multiprocessing.dummy import active_children
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple
from FlappyAgents.abst_flappy_agent import FlappyAgent
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor


class newDeepQAgent(FlappyAgent):
    def __init__(self):
        self.learning_rate: float = 0.1  # alpha
        self.discount: float = 1  # gamma
        self.epsilon: float = 0.1
        self.fig = None
        self.nbr_of_episodes = 0
        self.nbr_of_frames = 0

        self.model: MLPRegressor = MLPRegressor(hidden_layer_sizes=(100, 10),
                                                activation='logistic',
                                                learning_rate_init=0.1,
                                                random_state = 0)
        self.initialize_model()
        

    def get_q(self, state: Dict[str, int], action: int) -> float:        
        internal_state: Tuple[int, int, int, int] = self.state_to_internal_state(state)
        internal_state = internal_state + (action,)

        return self.model.predict(internal_state)

    def set_q(self, state: Dict[str, int], action: int, value: float) -> None:
        # Sets a value to Q(s,a) based on a state and action pair

        # internal_state: Tuple[int, int, int,
        #                       int] = self.state_to_internal_state(state)

        # if not (internal_state in self.q_values):
        #     self.q_values[internal_state] = dict()

        # self.q_values[internal_state][action] = value
        self.model.partial_fit([self.state_to_internal_state(state), action], value)

    def observe(self, iteration, s1: Dict[str, int], action: int, reward: float, s2: Dict[str, int], end: bool) -> None:
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """

        current_state_value: float = self.get_q(s1, action)

        if current_state_value is None:
            current_state_value = 0

        o_value: float = current_state_value + self.learning_rate * \
            (reward + self.discount * self.get_q(s2,
             self.action_with_max_value(s2)) - self.get_q(s1, action))

        # q_flap = self.get_q(s1, 1)
        # q_no_flap = self.get_q(s1, 0)

        self.set_q(s1, action, o_value)

        if end:
            self.nbr_of_episodes += 1

    def action_with_max_value(self, state: Dict[str, int]) -> int:

        s_0 = self.get_q(state, 0)
        s_1 = self.get_q(state, 1)

        # TODO
        # need to handle the case if there's no values for either s_0 or s_1

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

    def state_to_internal_state(self, state: Dict[str, int]) -> Tuple[int, int, int, int]:
        """ Normalizes the state in the range [-1,1]
        """

        player_y_normalized = 2 * (state['player_y'] - 0) / (512 - 0) - 1
        next_pipe_top_y_normalized = 2 * \
            (state['next_pipe_top_y'] - 0) / (512 - 0) - 1
        next_pipe_dist_to_player_normalized = 2 * \
            (state['next_pipe_dist_to_player'] - 0) / (288 - 0) - 1
        player_vel_normalized = 2 * (state['player_vel'] - -8) / (10 - -8) - 1

        return [player_y_normalized, next_pipe_top_y_normalized, next_pipe_dist_to_player_normalized, player_vel_normalized]

    def initialize_model(self):
        # self.q_values = dict()

        five_tuple = (0, 0, 0, 0, 0)

        # create an np array with 5 0's
        np_array = np.zeros(5).reshape(1, -1)
        
        self.model.fit(np_array, [0])