import numpy as np
from alg1 import greedy_naive_solution
from matrix import *
from itertools import accumulate


def local_search(w, n, constructive_func, save_iters=False):
    cur_solution = constructive_func(w, n)
    cur_objective = obj_function(w, cur_solution)

    n, num_cliques = len(w), len(cur_solution)
    m = initialize_matrix(w, cur_solution)

    if save_iters:
        it = 0
        deltas = [cur_objective]

    while True:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        max_delta = m[i, j]
        if max_delta == 0:
            break

        cur_clique = find_clique(cur_solution, i)

        # пересчет матрицы приращений
        m = modify_matrix(m, w, cur_solution, n, num_cliques, i, cur_clique, j)

        # обновление словаря клик
        #            вынести в отдельную функцию? или не стоит? что там с памятью...
        cur_solution[cur_clique].remove(i)

        needs_reindex = False
        if not cur_solution[cur_clique]:
            del cur_solution[cur_clique]
            needs_reindex = True
        if j == m.shape[1]:  # создаем новую клику
            new_idx = len(cur_solution)
            cur_solution[new_idx] = [i]
        else:
            cur_solution[j].append(i)

        if needs_reindex:
            # переиндексация ключей после удаления пустой клики
            old_keys = list(cur_solution.keys())
            for new_idx, old_key in enumerate(old_keys):
                cur_solution[new_idx] = cur_solution.pop(old_key)

        cur_objective += max_delta

        if save_iters:
            deltas.append(max_delta)
            it += 1

    return (cur_solution, cur_objective) if not save_iters else (cur_solution, cur_objective, deltas)


w1 = np.array([[ 0, -4, 18, -1, 12],
       [ 0,  0, -5,  6, -2],
       [ 0,  0,  0,  2, 17],
       [ 0,  0,  0,  0, 15],
       [ 0,  0,  0,  0,  0]])
n1 = 5

best_sol = local_search(w1, n1, greedy_naive_solution, save_iters=True)
print("лучшее разбиение на клики:", best_sol)
print(list(accumulate(best_sol[2])))
