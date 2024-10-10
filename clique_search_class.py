import copy
import numpy as np

from alg1 import greedy_naive_solution
from alg2 import vertex_deleter
from itertools import accumulate

import copy
import numpy as np


class CliqueSearch:
    def __init__(self, w, n, constructive_func, save_iters=False):
        self.w = w
        self.n = n
        self.constructive_func = constructive_func
        self.cliques = constructive_func(w, n)
        self.m = self.initialize_matrix(w, self.cliques)
        self.initial_sol = copy.deepcopy(self.cliques)
        if save_iters:
            self.deltas = []

    def reset_instance(self):
        # заново инициализирует поля
        # нужен для вызова нескольких способов решения на одном и том же экземпляре класса
        self.cliques = self.constructive_func(self.w, self.n)
        self.m = self.initialize_matrix(self.w, self.cliques)
        self.deltas = []

    def obj_function(self):
        total_weight = 0
        for clique in self.cliques.values():
            for i in clique:
                for j in clique:
                    if i != j and i < j:
                        total_weight += self.w[i][j]
        return total_weight

    def get_edge(self, i, j):
        return [i, j] if i < j else [j, i]

    def get_weights(self, v):
        if v == 0:
            return np.concatenate(([0], self.w[v, 1:]))

        left = self.w[:v, v]
        right = self.w[v, v + 1:] if v < self.n - 1 else []

        return np.concatenate((left, [0], right))

    def find_clique(self, i):
        for idx, clique in self.cliques.items():
            if i in clique:
                return idx

    def get_delta(self, cur_clique, new_clique, i):
        delta = 0

        for v in self.cliques[new_clique]:
            edge = self.get_edge(i, v)
            delta += self.w[edge[0], edge[1]]

        for v in self.cliques[cur_clique]:
            if v != i:
                edge = self.get_edge(i, v)
                delta -= self.w[edge[0], edge[1]]

        return delta

    def get_delta_dummy_clique(self, cur_clique, i):
        delta = 0
        for v in self.cliques[cur_clique]:
            if v != i:
                edge = self.get_edge(i, v)
                delta -= self.w[edge[0], edge[1]]
        return delta

    def initialize_matrix(self, w, cliques):
        n = len(w)
        num_cliques = len(cliques)

        m = np.zeros((n, num_cliques + 1))

        for i in range(n):
            my_clique = self.find_clique(i)

            for j in range(num_cliques):
                if my_clique != j:
                    m[i, j] = self.get_delta(my_clique, j, i)

            m[i, num_cliques] = self.get_delta_dummy_clique(my_clique, i)

        return m

    def update_cliques(self, i, cur_clique, new_clique):
        self.cliques[cur_clique].remove(i)

        needs_reindex = False
        if not self.cliques[cur_clique]:
            del self.cliques[cur_clique]
            needs_reindex = True

        if new_clique == len(self.cliques):
            self.cliques[new_clique] = [i]
        else:
            self.cliques[new_clique].append(i)

        if needs_reindex:
            self.cliques = {new_idx: clique for new_idx, clique in enumerate(self.cliques.values())}

    def modify_matrix(self, i, j, k):
        # i - индекс вершины, j - индекс клики "откуда", k - индекс клики "куда"
        cur_clique, new_clique = self.cliques[j], self.cliques[k]
        self.m[i] = [x - self.m[i, k] for x in self.m[i]]

        # изменение приращений клик "откуда" и "куда"
        #
        # сначала здесь избегала не только саму вершину, но и не изменяла ее старых и новых соседей по кликам,
        # потом поняла что если их не обходить здесь мы теряем информацию о том, в какой они были клике
        for v in range(self.n):
            if v != i:
                edge = self.get_edge(i, v)
                self.m[v, j] -= self.w[edge[0], edge[1]]
                if k != len(self.cliques):
                    self.m[v, k] += self.w[edge[0], edge[1]]

        # подшивание столбца
        if k == len(self.cliques):
            new_column = self.get_weights(i)
            for v in range(self.n):
                if v != i:
                    v_weights = self.get_weights(v)
                    v_weights[i] = 0
                    new_column[v] -= sum(v_weights)
            self.m = np.insert(self.m, len(self.cliques), new_column, axis=1)

        # убирание столбца
        if len(cur_clique) == 1:
            self.m = np.delete(self.m, j, axis=1)

        # пересчет строк соседей
        for v in cur_clique:
            if v != i:
                edge = self.get_edge(i, v)
                self.m[v] = [x + self.w[edge[0], edge[1]] for x in self.m[v]]

        for v in new_clique:
            edge = self.get_edge(i, v)
            self.m[v] = [x - self.w[edge[0], edge[1]] for x in self.m[v]]

    def local_search(self, save_iters=False):
        cur_objective = self.obj_function()
        if save_iters:
            it = 0
            self.deltas.append(cur_objective)

        while True:
            i, j = np.unravel_index(np.argmax(self.m), self.m.shape)
            max_delta = self.m[i, j]

            if max_delta <= 0:
                break

            cur_clique = self.find_clique(i)
            self.modify_matrix(i, cur_clique, j)
            self.update_cliques(i, cur_clique, j)

            cur_objective += max_delta
            # cur_objective = self.obj_function()
            if save_iters:
                self.deltas.append(max_delta)  # .append(cur_objective)
                it += 1

        return (self.cliques, cur_objective) if not save_iters else (self.cliques, cur_objective, self.deltas)

    def tabu_search(self, max_k, len_tl, save_iters=False):
        cur_objective = self.obj_function()

        tabu_lst, k = [], 0
        best_solution, best_objective = self.cliques, cur_objective

        if save_iters:
            it = 0
            self.deltas.append(cur_objective)

        while k < max_k:
            i, j = np.unravel_index(np.argmax(self.m), self.m.shape)
            max_delta = self.m[i, j]

            if max_delta == 0:
                break

            cur_clique = self.find_clique(i)
            candidate = (i, cur_clique, j)
            if candidate in tabu_lst:
                continue

            self.modify_matrix(i, cur_clique, j)
            self.update_cliques(i, cur_clique, j)

            tabu_lst.append(candidate)
            if len(tabu_lst) > len_tl:
                tabu_lst.pop(0)

            cur_objective += max_delta
            # cur_objective = self.obj_function()

            if cur_objective > best_objective:
                best_solution, best_objective = self.cliques, cur_objective
                k = 0
            else:
                k += 1

            if save_iters:
                self.deltas.append(max_delta)  # .append(cur_objective)
                it += 1

        return (best_solution, best_objective) if not save_iters else (best_solution, best_objective, self.deltas)

test_n = 20
test_w = np.array([
    [9.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 3.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 1.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, 10.0, 8.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, 10.0, 5.0, -1e+20, 1.0, 6.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 8.0, -1e+20, -1e+20, 6.0, -1e+20, 8.0, 7.0],
    [-1e+20, 8.0, -1e+20, 3.0, -1e+20, -1e+20, 9.0, -1e+20, -1e+20, -1e+20, 5.0, 5.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, 1.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, 6.0, -1e+20, -1e+20, -1e+20, 1.0, -1e+20, -1e+20, -1e+20, 8.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 10.0, 10.0],
    [3.0, -1e+20, -1e+20, 9.0, -1e+20, 1.0, 1.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 5.0, -1e+20, -1e+20, -1e+20, 9.0, -1e+20, 9.0],
    [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 10.0, -1e+20, -1e+20, 4.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 4.0, -1e+20],
    [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 10.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 1.0, 4.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 5.0, -1e+20, 7.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, -1e+20, 5.0, -1e+20, 8.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 7.0, -1e+20, 3.0, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, -1e+20, 5.0, -1e+20, -1e+20, -1e+20, 4.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 5.0, -1e+20, -1e+20, -1e+20, 9.0, 4.0, -1e+20],
    [1.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 5.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, 8.0, -1e+20, -1e+20, -1e+20, 5.0, -1e+20, 1.0, -1e+20, 7.0, 5.0, -1e+20, -1e+20, 5.0, 9.0, 2.0, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 4.0, 7.0, -1e+20, -1e+20, -1e+20, 5.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 3.0, -1e+20, -1e+20, 9.0, -1e+20, -1e+20, -1e+20, 10.0, -1e+20, -1e+20],
    [-1e+20, -1e+20, 6.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 2.0, -1e+20, -1e+20, -1e+20, 3.0, -1e+20, -1e+20],
    [-1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, 9.0, -1e+20, -1e+20, -1e+20, -1e+20, 9.0, -1e+20, -1e+20, -1e+20, 10.0, 3.0, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, 8.0, -1e+20, -1e+20, 10.0, -1e+20, 4.0, -1e+20, -1e+20, -1e+20, 4.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20],
    [-1e+20, -1e+20, 7.0, -1e+20, -1e+20, 10.0, 9.0, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20, -1e+20]
])

search_w = CliqueSearch(test_w, test_n, greedy_naive_solution, save_iters=True)
best_sol_loc_search, best_obj_loc_search, deltas_loc_search = search_w.local_search(save_iters=True)
search_w.reset_instance()
best_sol_tabu_search, best_obj_tabu_search, deltas_tabu_search = search_w.tabu_search(max_k=100, len_tl=5, save_iters=True)


def obj_function(w, cliques):
    total_weight = 0

    for clique in cliques.values():
        for i in clique:
            for j in clique:
                if i != j and i < j:
                    total_weight += w[i][j]

    return total_weight

print(f"best_sol_ls {best_sol_loc_search},\n best_obj_ls {best_obj_loc_search},\n deltas_ls {deltas_loc_search}")
print(f"\nobj_function(test_w, best_sol_loc_search) {obj_function(test_w, best_sol_loc_search)}\n\n")

print(f"best_sol_ls {best_sol_tabu_search},\n best_obj_ls {best_obj_tabu_search},\n deltas_ls {deltas_tabu_search}")
print(f"\nobj_function(test_w, best_sol_tabu_search) {obj_function(test_w, best_sol_tabu_search)}\n\n")
