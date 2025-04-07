import itertools
import pandas as pd

# Exchange rate matrix
currencies = ['Snowball', 'Pizza', 'Nugget', 'Shell']
exchange_rates = {
    'Snowball': [1, 1.45, 0.52, 0.72],
    'Pizza':    [0.7, 1, 0.31, 0.48],
    'Nugget':   [1.95, 3.1, 1, 1.49],
    'Shell':    [1.34, 1.98, 0.64, 1]
}

# Create DataFrame for easy access
df = pd.DataFrame(exchange_rates, index=currencies)

# Function to calculate value after a path of trades
def calculate_path_value(path, start_amount=1):
    value = start_amount
    for i in range(len(path) - 1):
        from_curr = path[i]
        to_curr = path[i + 1]
        value *= df[from_curr][to_curr] #####NOTE gpt fucked this line up, might cook some people
    return value

# Generate all 5-step paths starting and ending with "Shell"
profitable_paths = []
for path in itertools.product(currencies, repeat=4):
    full_path = ['Shell'] + list(path) + ['Shell']
    value = calculate_path_value(full_path)
    if value > 1.0:
        profitable_paths.append((full_path, value))

# Sort by most profitable
profitable_paths.sort(key=lambda x: -x[1])
top_paths = profitable_paths[:5]  # Get top 5 arbitrage paths

import pandas as pd
from tabulate import tabulate


print(tabulate(top_paths, headers='keys', tablefmt='pretty'))

