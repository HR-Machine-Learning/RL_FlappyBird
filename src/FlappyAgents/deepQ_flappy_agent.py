from multiprocessing.dummy import active_children
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple
from neural_network import NeuralNetwork
from FlappyAgents.abst_flappy_agent import FlappyAgent
import random


class NeuralNetworkAgent(FlappyAgent):
    def __init__(self):
        self.learning_rate: float = 0.1  # alpha
        self.discount: float = 1  # gamma
        self.epsilon: float = 0.1
        self.fig = None
        self.nbr_of_episodes: int = 0
        self.nbr_of_frames: int = 0
        self.replay_buffer: np.ndarray = np.array([])
        self.neural_network: NeuralNetwork = NeuralNetwork((100, 10), 'logistic', 0.1)

    def observe(self, iteration: int, s1: Dict[str, int], action: int, reward: float, s2: Dict[str, int], end: bool) -> None:
        current_state: Tuple[int, int, int, int] = self.state_to_internal_state(s1)
        if iteration >= 1001:
            current_state_value: int = self.neural_network.predict_next_state(current_state)
        else:
            current_state_value: int = random.randint(0, 1)

        # TODO replace calls to q with neural netwok
        o_value: float = reward + self.discount * self.action_with_max_value(s2)

        np.append(self.replay_buffer, o_value)

        if self.replay_buffer.size == 1000:
            self.neural_network.partial_training(self.replay_buffer)
            self.replay_buffer: np.ndarray = np.array([]) 

        if end:
            self.nbr_of_episodes += 1

    def action_with_max_value(self, state: Tuple[int, int, int, int]) -> int:
        if self.replay_buffer.size < 1001:
            return random.randint(0,1)
        else:
            self.neural_network.predict_next_state(state)

    def policy(self, state: Dict[str, int]) -> int:
        return self.action_with_max_value(self.neural_network.predict_next_state(self.state_to_internal_state(state)))

    def training_policy(self, state: Dict[str, int]) -> int:
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        greedy: bool = np.random.choice(
            [False, True], p=[self.epsilon, 1 - self.epsilon])

        if greedy:
            action: int = self.action_with_max_value(self.state_to_internal_state(state))
        else:
            action: int = random.randint(0, 1)

        return action
