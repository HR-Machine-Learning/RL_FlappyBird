from multiprocessing.dummy import active_children
from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple
from neural_network import NeuralNetwork
from FlappyAgents.abst_flappy_agent import FlappyAgent


class NeuralNetworkAgent(FlappyAgent):
    def __init__(self):
        self.learning_rate: float = 0.1  # alpha
        self.discount: float = 1  # gamma
        self.epsilon: float = 0.1
        self.fig = None
        self.nbr_of_episodes = 0
        self.nbr_of_frames = 0
        self.replay_buffer = np.empty(shape=1000)
        self.neural_network = NeuralNetwork((100, 10), 'logistic', 0.1)

    def observe(self, iteration, s1, action, reward, s2, end):
        current_state = self.state_to_internal_state(s1)
        if iteration >= 1001:
            current_state_value = self.neural_network.predict(current_state)
        else:
            current_state_value = random.randint(0, 1)

        o_value: float = current_state_value + self.learning_rate * \
            (reward + self.discount * self.get_q(s2,
             self.action_with_max_value(s2)) - self.get_q(s1, action))

        self.replay_buffer.append(o_value)

        if self.replay_buffer.size == 1000:
            self.neural_network.partial_training(self.replay_buffer)
            self.replay_buffer = np.empty(shape=1000)

        if end:
            self.nbr_of_episodes += 1

    def action_with_max_value(self, state: Dict[str, int]) -> int:
        return random.randint(0, 1)  # TODO tmp solution to make it runnable

    def policy(self, state) -> int:
        return self.action_with_max_value(self.neural_network.predict_next_state(self.state_to_internal_state(state)))

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        greedy: bool = np.random.choice(
            [False, True], p=[self.epsilon, 1 - self.epsilon])

        if greedy:
            action = self.action_with_max_value(state)
        else:
            action = random.randint(0, 1)

        return action
