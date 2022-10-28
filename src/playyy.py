from multiprocessing.dummy import active_children
import os
import time
from ple.games.flappybird import FlappyBird
from ple import PLE
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple
from neural_network import NeuralNetwork

from FlappyAgents.abst_flappy_agent import FlappyAgent
from FlappyAgents.deepQ_flappy_agent import NeuralNetworkAgent
from FlappyAgents.qlearning_flappy_agent import QlearningAgent

def main() -> None:
    # SETUP
    iterations: int = 4000
    #agent: FlappyAgent = QlearningAgent()
    agent: FlappyAgent = NeuralNetworkAgent()
    filestr: str = 'results/qvalues_' + str(iterations)

    train(iterations, agent)
    #write(agent, filestr)

    #vals = read("results/qvalues_4000")
    #print(filestr)
    #vals = read(filestr)
    #agent.q_values = vals

    #print(len(agent.q_values))

    #gent.plot("pi")
    run_game(10, agent)
    #agent.plot("pi")

def run_game(nb_episodes: int, agent: FlappyAgent) -> None:
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = agent.reward_values()

    # env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
    #         reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True
    env: PLE = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
                   reward_values=reward_values)
    env.init()

    score: int = 0

    while nb_episodes > 0:
        action: int = agent.policy(env.game.getGameState())

        # s1: Dict[str, int] = env.getGameState()

        # step the environment
        thing: int = env.act(env.getActionSet()[action])

        reward: int = thing

        end: bool = env.game_over()
        # s2 = env.getGameState()
        # agent.observe(s1, action, reward, s2, end)

        score += reward

        # reset the environment if the game is over
        if end:
            env.reset_game()
            nb_episodes -= 1
            score = 0


def train(nb_episodes: int, agent: FlappyAgent) -> None:
    reward_values: Dict[str, float] = agent.reward_values()

    env: PLE = PLE(FlappyBird(), 
                   fps = 30, 
                   display_screen = False, 
                   force_fps = True, 
                   rng = None,
                   reward_values = reward_values)
    env.init()

    score: int = 0
    prev_time = 0
    
    iteration = 0
    while nb_episodes > 0:
        # pick an action
        state: Dict[str, int] = env.game.getGameState()
        action: int = agent.training_policy(state)

        reward: int = env.act(env.getActionSet()[action])

        newState = env.game.getGameState()
        agent.observe(iteration, state, action, reward, newState, env.game_over())

        score += reward

        # reset the environment if the game is over
        if env.game_over():
            # print("-------------")
            # print("number of q values", len(agent.q_values))
            # print("NEW EPISODE:", nb_episodes)
            # print("score for this episode: %d" % score)
            if nb_episodes % 1000 == 0:
                curr_time = time.time()
                print(curr_time - prev_time, ":")
                      #nb_episodes, "-", len(agent.q_values))
                prev_time = curr_time
            env.reset_game()
            nb_episodes -= 1
            score = 0
        
        iteration += 1


def write(agent, filestr):

    # WRITE
    if not os.path.exists("results"):
        os.mkdir("results")

    while os.path.exists(filestr):
        filestr = + "_new"

    filestr += ".csv"

    w = csv.writer(open(filestr, "w"))
    for key, val in agent.q_values.items():
        zero = ""
        one = ""
        if val != None:
            if 0 in val:
                zero = val[0]
            if 1 in val:
                one = val[1]
        a, b, c, d = key
        w.writerow([a, b, c, d, zero, one])


def read(filestr):
    qvals = dict()
    # READ
    with open(filestr + ".csv", mode='r') as file:
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        for line in csvFile:
            if line == []:
                continue
            a, b, c, d, zero, one = line

            t = dict()

            if zero != "":
                t[0] = float(zero)
            if one != "":
                t[1] = float(one)

            qvals[(int(a), int(b), int(c), float(d))] = t

    return qvals

main()
