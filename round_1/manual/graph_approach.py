# from gpt

import networkx as nx
import math

# Step 1: Set up FX rates (with arbitrage opportunity)
rates = {
    "snowballs": {
        "snowballs": 1,
        "pizzas": 1.45,
        "nuggets": 0.52,
        "shells": 0.72
    },
    "pizzas": {
        "snowballs": 0.7,
        "pizzas": 1,
        "nuggets": 0.31,
        "shells": 0.48
    },
    "nuggets": {
        "snowballs": 1.95,
        "pizzas": 3.1,
        "nuggets": 1,
        "shells": 1.49
    },
    "shells": {
        "snowballs": 1.34,
        "pizzas": 1.98,
        "nuggets": 0.64,
        "shells": 1
    }
}

# Convert nested dict into flat fx_rates format
fx_rates = {}
for src, targets in rates.items():
    for dst, rate in targets.items():
        if src != dst:  # Skip self-exchanges unless needed
            fx_rates[(src, dst)] = rate

# Create directed graph with -log(rate) as weights
G = nx.DiGraph()
for (src, dst), rate in fx_rates.items():
    weight = -math.log(rate)
    G.add_edge(src, dst, weight=weight, rate=rate)

# Step 2: Bellman-Ford with cycle detection
def bellman_ford(graph, source):
    dist = {node: float('inf') for node in graph.nodes}
    pred = {node: None for node in graph.nodes}
    dist[source] = 0

    for _ in range(len(graph.nodes) - 1):
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                pred[v] = u

    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        if dist[u] + weight < dist[v]:
            return dist, pred, (u, v)  # Negative cycle found

    return dist, pred, None

# Step 3: Extract cycle using predecessor
def extract_cycle(predecessor, start):
    visited = set()
    cycle = []
    node = start

    for _ in range(len(predecessor)):
        node = predecessor[node]

    cycle_start = node
    while node not in cycle:
        cycle.append(node)
        node = predecessor[node]

    cycle.append(node)
    cycle.reverse()
    return cycle

# Step 4: Search for best arbitrage cycle
best_cycle = None
best_weight = float('inf')

for src in ['shells']:
    dist, pred, cycle_edge = bellman_ford(G, src)
    if cycle_edge:
        _, v = cycle_edge
        cycle = extract_cycle(pred, v)
        if cycle[0] == 'shells' and cycle[-1] == 'shells':
            cycle_weight = sum(G[u][v]['weight'] for u, v in zip(cycle, cycle[1:]))
            if cycle_weight < best_weight:
                best_weight = cycle_weight
                best_cycle = cycle


# Step 5: Convert cycle to readable trades and profit estimation
trade_path = []
profit_ratio = 1.0

if best_cycle:
    for u, v in zip(best_cycle, best_cycle[1:]):
        rate = G[u][v]['rate']
        trade_path.append(f"{u} → {v} @ {rate}")
        profit_ratio *= rate

    initial_amount = 1000
    final_amount = initial_amount * profit_ratio
    profit = final_amount - initial_amount
else:
    trade_path = ["No arbitrage opportunity found."]
    profit = 0
    final_amount = 0

# Print results
print("\n".join(trade_path))
print(f"\nProfit from arbitrage: ${profit:.2f}")
print(f"Final amount: ${final_amount:.2f}")