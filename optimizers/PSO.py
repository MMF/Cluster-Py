# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016

@author: Hossam Faris
"""

import random
import numpy
import math
import time

class solution:
    def __init__(self):
        self.leader_fitness = 0
        self.leader_solution=[]
        self.leader_score = 0
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
        self.func_evals = 0

def PSO(objf, lb, ub, dim, PopSize, iters, verbose=False, plt_func=None, stopping_func=None):
    # PSO parameters
    best_score = 0

    #    dim=30
    #    iters=200
    Vmax = 4
    #    PopSize=50     #population size
    wMax = 0.9
    wMin = 0.1
    c1 = 2
    c2 = 2
    #    lb=-10
    #    ub=10
    #

    # convert lower_bound, upper_bound to array
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [lb for _ in range(dim)]

    s = solution()

    ######################## Initializations

    vel = numpy.zeros((PopSize, dim))

    pBestScore = numpy.zeros(PopSize)
    pBestScore.fill(float("inf"))

    pBest = numpy.zeros((PopSize, dim))
    gBest = numpy.zeros(dim)

    gBestScore = float("inf")

    #pos = numpy.random.uniform(0, 1, (PopSize, dim)) * (ub - lb) + lb

    pos = []
    for i in range(PopSize):
        sol = []
        for d in range(dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        pos.append(sol)

    pos = numpy.array(pos)

    convergence_curve = numpy.zeros(iters)

    ############################################
    #print("PSO is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):

        if stopping_func is not None and stopping_func(gBestScore, pos, l):
            break

        for i in range(0, PopSize):
            # pos[i,:]=checkBounds(pos[i,:],lb,ub)
            pos[i, :] = numpy.clip(pos[i, :], lb, ub)
            # Calculate objective function for each particle
            fitness = objf(pos[i, :])
            s.func_evals += 1

            if (pBestScore[i] > fitness):
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            if (gBestScore > fitness):
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iters);

        for i in range(0, PopSize):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = w * vel[i, j] + c1 * r1 * (pBest[i, j] - pos[i, j]) + c2 * r2 * (gBest[j] - pos[i, j])

                if (vel[i, j] > Vmax):
                    vel[i, j] = Vmax

                if (vel[i, j] < -Vmax):
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore

        if plt_func is not None:
            plt_func(l, pos, gBestScore)

        if (l % 10 == 0 and verbose):
            print('Iter: ' + str(l + 1) + ', Fitness: ' + str(gBestScore) + ', Pos= ' + str(gBest));
            #print(['Iteration: ' + str(l + 1) + ', Fitness: ' + str(gBestScore) + ', Score: ' + str(s.leader_score)]);

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "PSO"
    s.objfname = objf.__name__
    s.leader_fitness = gBestScore
    s.leader_score = best_score
    s.leader_solution = gBest

    return s

