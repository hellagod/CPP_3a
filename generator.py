import json

import numpy as np
import random


def generate(n):
    matrix = np.full((n, n), -1e20)

    for i in range(n):
        for j in range(i + 1):
            if random.random() < 0.9:
                value = random.randint(-1000, 3000)
                matrix[i][j] = value

    for i in range(n):
        for j in range(i + 1):
            if i != j:
                matrix[j][i] = matrix[i][j]

    return {
        'n': n,
        'weights': matrix.tolist()
    }


with open(f'dataset.json', 'w', encoding='utf-8') as file:
    json.dump(generate(100), file, ensure_ascii=False)