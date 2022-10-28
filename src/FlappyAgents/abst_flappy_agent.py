import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple
import abc


class FlappyAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    def reward_values(self) -> Dict[str, float]:
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def translate_int(self, value: int, leftMin: int, leftMax: int, rightMin: int, rightMax: int) -> int:
        # Figure out how 'wide' each range is
        leftSpan: int = leftMax - leftMin
        rightSpan: int = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled: float = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return max(min(floor(rightMin + (valueScaled * rightSpan)), rightMax), rightMin)

    def state_to_internal_state(self, state: Dict[str, int]) -> Tuple[int, int, int, int]:
        """ Maps a game state to the discretized version of it.
        """

        player_y: int = state['player_y']
        next_pipe_top_y: int = state['next_pipe_top_y']
        next_pipe_dist_to_player: int = state['next_pipe_dist_to_player']
        player_vel: int = state['player_vel']

        player_y: int = self.translate_int(player_y, 0, 512, 0, 15)
        next_pipe_top_y: int = self.translate_int(
            next_pipe_top_y, 0, 512, 0, 15)
        next_pipe_dist_to_player: int = self.translate_int(
            next_pipe_dist_to_player, 0, 288, 0, 15)

        if not (0 <= player_y <= 15 and 0 <= next_pipe_top_y <= 15 and 0 <= next_pipe_dist_to_player <= 15):
            raise Exception

        return (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)

    @abc.abstractmethod
    def observe(self, s1, action, reward, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        pass

    @abc.abstractmethod
    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        pass

    @abc.abstractmethod
    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        pass

    def plot(self, what):
        data = [k + tuple(self.q_values[k]) for k in self.q_values.keys()]
        if self.fig == None:
            self.fig = plt.figure()
        else:
            plt.figure(self.fig.number)
        self.fig.show()
        self.fig.canvas.draw()
        self.fig, ax = plt.subplots(figsize=(20, 20))
        df = pd.DataFrame(data=data, columns=('next_pipe_top_y', 'player_y',
                          'player_vel', 'next_pipe_dist_to_player', 'q_flap', 'q_noop'))
        df['delta_y'] = df['player_y'] - df['next_pipe_top_y']
        df['v'] = df[['q_noop', 'q_flap']].max(axis=1)
        df['pi'] = (df[['q_noop', 'q_flap']].idxmax(axis=1) == 'q_flap')*1
        selected_data = df.groupby(
            ['delta_y', 'next_pipe_dist_to_player'], as_index=False).mean()
        plt.clf()
        with sns.axes_style("white"):
            if what in ('q_flap', 'q_noop', 'v'):
                ax = sns.heatmap(selected_data.pivot('delta_y', 'next_pipe_dist_to_player',
                                 what), vmin=-5, vmax=5, cmap='coolwarm', annot=True, fmt='.2f')
            elif what == 'pi':
                ax = sns.heatmap(selected_data.pivot(
                    'delta_y', 'next_pipe_dist_to_player', 'pi'), vmin=0, vmax=1, cmap='coolwarm')
            ax.invert_xaxis()
            ax.set_title(what + ' after ' + str(self.nbr_of_frames) +
                         ' frames / ' + str(self.nbr_of_episodes) + ' episodes')
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.1)
        self.fig.savefig(what + '_' + 'plot.jpg', bbox_inches='tight', dpi=150)
