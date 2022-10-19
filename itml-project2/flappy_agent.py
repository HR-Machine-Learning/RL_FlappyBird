from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here

        self.discount_factor = 1 
        self.alpha = 0.7
        self.q = {}  # q-table[state][action]

        return
    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        # return {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, action, reward, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation

        
        # self.q[s1][action] = self.q[s1][action] + self.alpha * (reward + self.discount_factor * np.max(self.q[s2, :]) - self.q[s1, action])
        
        if self.q[s1][action] != None:
            self.q[s1][action] = (1 - self.alpha) * (self.q[s1][action]) + self.alpha * (reward + self.discount_factor * max(self.q[s2][0:2]))
        

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.

        # Exploration vs Explotiation?!

        return random.randint(0, 1) 

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.

        print(self.q)

        return random.randint(0, 1) 

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    # reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None,
            reward_values = reward_values)
    # env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
    #         reward_values = reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        # action = agent.policy(env.game.getGameState())

        action = agent.tranining_policy()  

        s1 = env.getGameState()

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        end = env.game_over()
        s2 = env.getGameState()
        agent.observe(s1, action, reward, s2, end)

        score += reward
        
        # reset the environment if the game is over
        if end:
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # let the agent observe the current state transition
        newState = env.game.getGameState()
        agent.observe(state, action, reward, newState, env.game_over())

        score += reward
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

agent = FlappyAgent()
train(1, agent)
