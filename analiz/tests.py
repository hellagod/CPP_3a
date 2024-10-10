import json

import numpy as np

from alg1 import greedy_naive_solution
from alg2 import vertex_deleter_solution
from clique_search_class import CliqueSearch
from generator import generate
from linear_model import CPPSoftModel
from utils import clique_weight
from tqdm import tqdm

ns = [30] * 50

ws = map(generate, ns)

tests = []

for w in tqdm(ws):
    try:
        cpp_model = CPPSoftModel(**w, soft=False)
        cpp_model.solve()
        output = cpp_model.output()
        search_w = CliqueSearch(np.asarray(w['weights']), w['n'], greedy_naive_solution, save_iters=True)
        best_sol_loc_search, best_obj_loc_search, deltas_loc_search = search_w.local_search(save_iters=True)
        search_w.reset_instance()
        best_sol_tabu_search, best_obj_tabu_search, deltas_tabu_search = search_w.tabu_search(max_k=100, len_tl=5,
                                                                                              save_iters=True)
        # search_w = CliqueSearch(np.asarray(w['weights']), w['n'], lambda w, n : dict(zip(range(n), list(map(lambda x: [x], range(n))))), save_iters=True)
        # best_sol_loc_search_1, best_obj_loc_search, deltas_loc_search = search_w.local_search(save_iters=True)
        # search_w.reset_instance()
        # best_sol_tabu_search_1, best_obj_tabu_search, deltas_tabu_search = search_w.tabu_search(max_k=100, len_tl=5,
        #                                                                                       save_iters=True)

        tests.append([output['f'],
                      clique_weight(greedy_naive_solution(**w).values(), w['weights']),
                      clique_weight(vertex_deleter_solution(**w).values(), w['weights']),
                      clique_weight(best_sol_tabu_search.values(), w['weights']),
                      clique_weight(best_sol_loc_search.values(), w['weights']),
                      # clique_weight(best_sol_tabu_search_1.values(), w['weights']),
                      # clique_weight(best_sol_loc_search_1.values(), w['weights']),
                      ])
    except Exception as e:
        print(e)

with open('solutions.json', 'w', encoding='utf-8') as file:
        json.dump(tests, file, ensure_ascii=False)