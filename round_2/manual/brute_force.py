import numpy as np

def game_theory(percent_smart, weights, bool_print = True): # play around with to get a feel for it. As this approaches 100, everyone becomes rational and we converge to nash

    base = 10000

    pool_of_players = 100
    total_pool = 1

    while pool_of_players > .0001:
        adj_weights = weights / total_pool
        adj_reward = rewards / (inhabitants + adj_weights) * base

        #first crate
        weights[adj_reward == max(adj_reward)] += pool_of_players * (100 - percent_smart) / 100

        #second crate
        temp = adj_reward.copy()
        temp[adj_reward == max(adj_reward)] = 0
        temp -= 50000
        if (temp > 0).any():
            weights[temp == max(temp)] += pool_of_players * (100 - percent_smart) / 100
            total_pool += pool_of_players / 100 * (100 - percent_smart) / 100

        #decrease original pool
        pool_of_players -= pool_of_players * (100 - percent_smart) / 100

    weights /= total_pool

    if bool_print:
        for weight, reward, inhabitant in zip(weights, rewards, inhabitants):
            print(f"We predict {weight:.5f} percent of people will go to container {reward} to get {(reward / (weight + inhabitant) * base):.7f} shells")

        print(sum(weights))

    return weights

rewards = np.array([90, 89, 80, 73, 50, 37, 31, 20, 17, 10])
inhabitants = np.array([10, 8, 6, 4, 4, 3, 2, 2, 1, 1])
base_weights = np.array([0 for x in range(10)], dtype=float)

total_weights = []

for i in range(40, 90):
    total_weights.append(game_theory(i, base_weights.copy(), False))

import matplotlib.pyplot as plt

best = max(rewards / (inhabitants + np.mean(total_weights, axis = 0)))
plt.bar([str(f"{r}\n{(r / (i + w) * 10000):.5f}") for r, i, w in zip(rewards, inhabitants, np.mean(total_weights, axis = 0))], np.mean(total_weights, axis = 0))
plt.show()