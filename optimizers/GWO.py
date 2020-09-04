# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

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

def GWO(objf,lb,ub,dim,solutions_count,Max_iter, verbose=True, stopping_func=None, plt_func=None):
    # convert lower_bound, upper_bound to array
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]

    # alpha
    alpha_pos = []
    alpha_fitness=float("inf")

    # beta
    beta_pos = []
    beta_fitness=float("inf")
    
    # delta
    delta_pos = []
    delta_fitness=float("inf")
    
    #Initialize the positions of search agents
    Positions = []
    for s in range(solutions_count):
        sol = []
        for d in range(dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        Positions.append(sol)

    Positions = numpy.array(Positions)

    # solution
    s=solution()

    # Main loop
    for l in range(0,Max_iter):
        # should i stop ?
        if stopping_func is not None and stopping_func(alpha_fitness, alpha_pos, l):
            break

        # calc 3 leaders (alpha, beta, delta)
        for i in range(0,solutions_count):
            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])
            
            # Update Alpha, Beta, and Delta
            if fitness<alpha_fitness :
                alpha_fitness=fitness; # Update alpha
                alpha_pos = Positions[i, :].copy()
            
            
            if (fitness>alpha_fitness and fitness<beta_fitness ):
                beta_fitness=fitness  # Update beta
                beta_pos = Positions[i, :].copy()
            
            
            if (fitness>alpha_fitness and fitness>beta_fitness and fitness<delta_fitness): 
                delta_fitness=fitness # Update delta
                delta_pos = Positions[i, :].copy()
            
        #if plt_func is not None:
        #    plt_func(l, Positions, alpha_fitness)

        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,solutions_count):
            # for each dimension
            for j in range (0,dim):
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1* alpha_pos[j]-Positions[i, j]); # Equation (3.5)-part 1
                X1=alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2 * beta_pos[j]-Positions[i, j]); # Equation (3.5)-part 2
                X2=beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3 * delta_pos[j] -Positions[i, j]); # Equation (3.5)-part 3
                X3=delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3
                
                Positions[i, j]=(X1+X2+X3)/3  # Equation (3.7)

        if (l%10==0 and verbose):
            log = 'iteration ' + str(l) + ' ,the best fitness: ' + str(alpha_fitness)
            print(log)
    
    s.optimizer="GWO"
    s.objfname=objf.__name__
    s.leader_fitness = alpha_fitness
    s.leader_solution = alpha_pos

    return s
    

