import pandas as pd
import matplotlib.pyplot as plt

# List of price CSV files for different days
price_files = [
    "../data/prices_round_1_day_-1.csv",
    "../data/prices_round_1_day_-2.csv",
    "../data/prices_round_1_day_0.csv"
]

# Create an empty list to store DataFrames from each file.
df_list = []

# Loop through each file, read it, and filter for Rainforest Resin
for file in price_files:
    # Assuming columns are separated by semicolons
    df = pd.read_csv(file, delimiter=';')
    
    # Convert key columns to numeric (errors set as NaN if conversion fails)
    df['bid_price_1'] = pd.to_numeric(df['bid_price_1'], errors='coerce')
    df['ask_price_1'] = pd.to_numeric(df['ask_price_1'], errors='coerce')
    df['mid_price']   = pd.to_numeric(df['mid_price'], errors='coerce')
    
    # Filter the DataFrame to keep only rows for Rainforest Resin
    # This assumes the 'product' column indicates the tradable item.
    df = df[df['product'] == "RAINFOREST_RESIN"]
    
    df_list.append(df)

# Concatenate all DataFrames into one for overall analysis.
prices_df = pd.concat(df_list, ignore_index=True)

# Compute the spread as the difference between the best ask and bid.
prices_df['spread'] = prices_df['ask_price_1'] - prices_df['bid_price_1']

# Display basic spread statistics
print("Spread Statistics for Rainforest Resin:")
print(prices_df['spread'].describe())

# Plot a histogram of the spread distribution
plt.figure(figsize=(8,6))
plt.hist(prices_df['spread'].dropna(), bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title("Distribution of Spreads for Rainforest Resin")
plt.xlabel("Spread (ask_price_1 - bid_price_1)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Plot spread over time using the timestamp column.
# This helps to observe how the spread evolves during a day.
plt.figure(figsize=(10,6))
plt.plot(prices_df['timestamp'], prices_df['spread'], marker='o', linestyle='-', color='green')
plt.title("Spread over Time for Rainforest Resin")
plt.xlabel("Timestamp")
plt.ylabel("Spread")
plt.grid(True)
plt.show()

# Optionally, analyze volume data at the best bid and ask.
prices_df['bid_volume_1'] = pd.to_numeric(prices_df['bid_volume_1'], errors='coerce')
prices_df['ask_volume_1'] = pd.to_numeric(prices_df['ask_volume_1'], errors='coerce')

print("Average Bid Volume:", prices_df['bid_volume_1'].mean())
print("Average Ask Volume:", prices_df['ask_volume_1'].mean())
