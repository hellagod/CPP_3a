import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

with open(f'solutions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data, columns=["Exact solution", "Naive greedy", "Naive v deleter", "soft tabu search", "soft local search", "tabu search", "local search"])

deviation_df = df.apply(lambda x: (x - df["Exact solution"]) / df["Exact solution"] * 100, axis=0)

plt.figure(figsize=(12, 8))

for column in deviation_df.columns[1:]:
    plt.plot(deviation_df.index, deviation_df[column], label=column, marker='o')

plt.title("Deviation of Algorithms from Exact Solution (%)")
plt.xlabel("Index")
plt.ylabel("Deviation from Exact Solution (%)")
plt.legend(title="Algorithms")
plt.grid(True)
plt.show()
