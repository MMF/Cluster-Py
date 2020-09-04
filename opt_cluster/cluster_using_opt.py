from optimizers.PSO import *
from optimizers.GWO import *
from optimizers.DE import *

from opt_cluster.objective_func import *

opt = DE(
    verbose=True,
    fitness_func=calc_fitness,
    population_size=30,
    individual_size=4,
    iters_count=100,
    lower_bound=lower_bounds,
    upper_bound=upper_bounds,
    mutation_factor=0.5,
    crossover_ratio=0.7
)


"""
opt = GWO(
    verbose=True,
    objf=calc_fitness,
    solutions_count=30,
    dim=4,
    Max_iter=100,
    lb=lower_bounds,
    ub=upper_bounds,
)
"""

"""
opt = PSO(
    verbose=True,
    objf=calc_fitness,
    lb=lower_bounds,
    ub=upper_bounds,
    iters=100,
    dim=4,
    PopSize=30
)
"""

print(opt.leader_fitness)
print(opt.leader_solution)