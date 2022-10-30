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
        #self.replay_buffer = np.array([[]])
        self.replay_buffer=[[0,0,0,0,[0,0]], [0,0,0,0,[0,0]]]
        self.model: MLPRegressor = MLPRegressor(hidden_layer_sizes=(100, 10),
                                                activation='logistic',
                                                learning_rate_init=0.1,
                                                random_state = 0)
                                            
    
        self.initialize_model()

        

    def get_q(self, state: Dict[str, int], action: int) -> float:        
        internal_state: Tuple[int, int, int, int] = self.state_to_internal_state(state)

        #print(internal_state)
        #print

        #state_action = (internal_state(0), internal_state(1), internal_state(2), internal_state(3), action)

        state_action = internal_state 
        np_state_action = np.array(state_action)

        return self.model.predict(np_state_action.reshape(1, -1))

    def set_q(self, state: Dict[str, int], action: int, value: float) -> None:
        # Sets a value to Q(s,a) based on a state and action pair

        internal_state: Tuple[int, int, int, int] = self.state_to_internal_state(state)

        # if not (internal_state in self.q_values):
        #     self.q_values[internal_state] = dict()

        # self.q_values[internal_state][action] = value
        #print(value)
        internal_state.append(value)
        #print(internal_state)
        #np_state_action = np.array([state_action])
        #print("state_action")
        #print(state_action)
        #print(self.replay_buffer)

        if len(self.replay_buffer) == 1000: 
            random.shuffle(self.replay_buffer)
            self.replay_buffer = self.replay_buffer[100:]
            random_states = self.replay_buffer[:100]   

            #with open("output_1.txt", "w") as txt_file:
            #    for line in random_states:
            #        txt_file.write(str(line) + "\n") # works with any number of elements in a line         

            states = [list(s[0:4]) for s in random_states]
            values = [s[4] for s in random_states]

            #with open("output.txt", "w") as txt_file:
            #    for line in values:
            #        txt_file.write(str(line) + "\n") # works with any number of elements in a line
            self.model.partial_fit(states, values)
        else: 
            self.replay_buffer.append(internal_state)


    def observe(self, iteration, s1: Dict[str, int], action: int, reward: float, s2: Dict[str, int], end: bool) -> None:
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """

        q_value = self.get_q(s2, self.action_with_max_value(s2))[0]
        best_q = q_value[self.action_with_max_value(s2)]
        
        updated_value: float = reward + self.discount * best_q 

        new_q_value = [0,0]    
        if best_q == 0:
            new_q_value[0] = updated_value
            new_q_value[1] = q_value[1]
        else: # if best_q is 1
            new_q_value[0] = q_value[0]
            new_q_value[1] = updated_value

        self.set_q(s1, action, new_q_value)

        if end:
            self.nbr_of_episodes += 1

    def action_with_max_value(self, state: Dict[str, int]) -> int:

        #s_0 = self.get_q(state, 0)
        #s_1 = self.get_q(state, 1)

        q = self.get_q(state,0) # dont care abput the action. this should return a pair of q-values, [no_flap, flap]
        
        q_0 = q[0][0]
        q_1 = q[0][1]

        if q_0 is None:
            q_0 = 0
        if q_1 is None:
            q_1 = 0
        if q_1 > q_0:
            return 1
        else:
            return 0

    def training_policy(self, state: Dict[str, int]) -> int:
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        greedy: bool = np.random.choice([False, True], p=[self.epsilon, 1 - self.epsilon])
        # greedy = False

        if greedy:
            action = self.action_with_max_value(state)
        else:
            action = random.randint(0, 1)

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
        #np_array = np.array([[0, 0, 0, 0, 0]])
        #np_array = np.append(np_array, [[0, 0, 0, 0, 0]], axis=0)

        np_array= [[1,2,3,4], [0,0,0,0]]

        self.model.fit(np_array, [[0,1],[0,1]])