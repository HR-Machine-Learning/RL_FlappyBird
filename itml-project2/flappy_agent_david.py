from multiprocessing.dummy import active_children
from ple.games.flappybird import FlappyBird
from ple import PLE
import random

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        self.q_values = {}
        self.learning_rate = 0.1 # alpha
        self.discount = 1 # gamma 
        # TODO add epsilon for greediness later? 
    
    def state_to_internal_state(self, state):
        # make a method that maps a game state to your discretized version of it, e.g., 
        # state_to_internal_state(state). 
        # The internal (discretized) version of a state should be kept in a data structure 
        # that can be easily used as a key in a dict (e.g., a tuple).

        # TODO discretize values?
        player_y = state['player_y']
        next_pipe_top_y = state['next_pipe_top_y']
        next_pipe_dist_to_player = state['next_pipe_dist_to_player']
        player_vel = state['player_vel']
        return (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)
    
    def get_q(self, state, action): 
        # Returns Q(s,a) based on a state and action pair
        # TODO handle edge cases. For example, 
        # what value do you return if a state or action does not appear in your dictionary 
        # because you have not learned a value for it yet?
    
        return self.q_values[self.state_to_internal_state(state)][action]
        # {(player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel), ?}}

    def set_q(self, state, action, value):
        # Sets a value to Q(s,a) based on a state and action pair
        self.q_values[self.state_to_internal_state(state)][action] = value

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation

        current_state = self.get_q(s1, a)

        if current_state is None:
            current_state = 0
        
        # Calculates return
        max_next_state = r + self.discount * self.max_value_a(s2) 

        # Updates Q(s, a) with a new value
        value = current_state + self.learning_rate * (max_next_state - current_state)
        self.set_q(s1, a, value)  

    def max_value_a(self, state):
        # TODO should return the action that yields the highest reward
        # state_action
        s_0 = self.get_q(state, 0)
        s_1 = self.get_q(state, 1)

        # TODO need to handle the case if there's no values for either s_0 or s_1

        if s_0 > s_1:
            return s_0
        else:
            return s_1

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
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
        return random.randint(0, 1) 

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward
        
        # reset the environment if the game is over
        if env.game_over():
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
train(3, agent)
