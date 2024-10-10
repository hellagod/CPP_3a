import random
from deap import base, creator, tools, algorithms
from clique_search_class import CliqueSearch
import numpy as np

from alg1 import greedy_naive_solution
from alg2 import vertex_deleter


def get_tabu_hyperparams_grid_search(w, n, max_lenTL, max_max_k, constructive_func):
    lenTL_vals = range(1, max_lenTL + 1, 2)
    max_k_vals = range(1, max_max_k + 1, 2)

    best_lenTL, best_max_k = None, None
    best_solution, best_obj = None, -np.inf

    for lenTL in lenTL_vals:
        for max_k in max_k_vals:
            search = CliqueSearch(w, n, constructive_func, save_iters=False)
            sol, obj = search.tabu_search(max_k, lenTL)
            if obj > best_obj:
                best_lenTL, best_max_k = lenTL, max_k
                best_solution, best_obj = sol, obj

    return best_lenTL, best_max_k, best_solution, best_obj


def get_tabu_hyperparams_ga(w, n, max_lenTL, max_max_k, population_size, constructive_func):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_lenTL", random.randint, 1, max_lenTL)
    toolbox.register("attr_max_k", random.randint, 1, max_max_k)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_lenTL, toolbox.attr_max_k), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def rank(individual):
        lenTL, max_k = individual
        search = CliqueSearch(w, n, constructive_func, save_iters=False)
        score = search.tabu_search(max_k, lenTL)[1]
        return (score,)

    toolbox.register("evaluate", rank)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(population, toolbox,
                        cxpb=0.5, mutpb=0.2, ngen=40, halloffame=hof, verbose=False)
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    return best_individual, best_fitness

n = 10
w = np.array([[2.0, 6.0, 7.0, -1e+20, -1e+20, -1e+20, -1e+20, 6.0, 3.0, -1e+20],
             [6.0, 10.0, -1e+20, -1e+20, -1e+20, 3.0, -1e+20, -1e+20, 3.0, -1e+20],
             [7.0, -1e+20, -1e+20, -1e+20, 6.0, 7.0, -1e+20, -1e+20, -1e+20, -1e+20],
             [-1e+20, -1e+20, -1e+20, 6.0, 3.0, -1e+20, -1e+20, 4.0, -1e+20, -1e+20],
             [-1e+20, -1e+20, 6.0, 3.0, -1e+20, 9.0, -1e+20, -1e+20, -1e+20, -1e+20],
             [-1e+20, 3.0, 7.0, -1e+20, 9.0, -1e+20, 10.0, 5.0, 4.0, 6.0],
             [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 10.0, -1e+20, -1e+20, 10.0, 10.0],
             [6.0, -1e+20, -1e+20, 4.0, -1e+20, 5.0, -1e+20, 6.0, 10.0, 1.0],
             [3.0, 3.0, -1e+20, -1e+20, -1e+20, 4.0, 10.0, 10.0, -1e+20, -1e+20],
             [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 6.0, 10.0, 1.0, -1e+20, 3.0]])

max_lenTL, max_max_k = 100, 100
best_lenTL, best_max_k, best_sol, best_obj = get_tabu_hyperparams_grid_search(w, n, max_lenTL, max_max_k, greedy_naive_solution)

print("grid search")
print("best lenTL:", best_lenTL)
print("best max_k:", best_max_k)
print("best objective:", best_obj)
print("\n\n")

population = 100
best_params, best_obj = get_tabu_hyperparams_ga(w, n, max_lenTL, max_max_k, population, greedy_naive_solution)
best_lenTL, best_max_k = best_params
print("gen algorithm")
print("best lenTL:", best_lenTL)
print("best max_k:", best_max_k)
print("best objective:", best_obj)

