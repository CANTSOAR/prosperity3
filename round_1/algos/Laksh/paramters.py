import itertools
import json

# Define the grid of parameters to test.
# For example, these parameters might be used in momentumandmean.py:
threshold_values = [0.5, 1.0, 1.5, 2.0]   # e.g., mean reversion threshold.
weight_values = [0.0, 0.5, 1.0]             # e.g., momentum weight.

# Prepare a list to hold our results.
results = []

# Loop over all parameter combinations.
for threshold, weight in itertools.product(threshold_values, weight_values):
    # Write parameters to a configuration file that momentumandmean.py will read.
    params = {
        "threshold": threshold,
        "weight": weight
    }
    # Save the parameters to 'params.txt' (or JSON if you prefer)
    with open("params.txt", "w") as f:
        json.dump(params, f)
    print("=" * 60)
    print(f"Parameters written: threshold = {threshold}, weight = {weight}")
    print("Please run the backtester using the command:")
    print("  prosperity3bt.exe .\\momentumandmean.py 1")
    print("Then, once the backtest completes, copy the line that starts with 'Total profit:'")
    profit_str = input("Paste the 'Total profit' line here: ").strip()
    
    # Try to extract the profit value from the input (assumes format: "Total profit: <profit_value>")
    try:
        profit = float(profit_str.split("Total profit:")[1].strip())
    except Exception as e:
        print("Could not parse profit value; defaulting profit to -inf for these parameters.")
        profit = float('-inf')
    
    results.append(((threshold, weight), profit))
    print(f"Recorded profit for threshold {threshold}, weight {weight}: {profit}\n")

# Once finished, sort the results by profit (highest profit first) and display the summary.
results.sort(key=lambda x: x[1], reverse=True)

print("\nParameter Optimization Results (sorted by Total profit):")
for (thr, wt), profit in results:
    print(f"Threshold: {thr}, Weight: {wt}  =>  Total profit: {profit}")

# Optionally, save results to a file.
with open("optimization_results.txt", "w") as f:
    for (thr, wt), profit in results:
        f.write(f"Threshold: {thr}, Weight: {wt}  =>  Total profit: {profit}\n")
