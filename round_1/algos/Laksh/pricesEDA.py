import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load Price Data ---

# List your price CSV files here.
price_files = [
    "../data/prices_round_1_day_-1.csv",
    "../data/prices_round_1_day_-2.csv",
    "../data/prices_round_1_day_0.csv"
]

# Read and combine the price files.
df_list = []
for file in price_files:
    df = pd.read_csv(file, delimiter=';')
    df_list.append(df)

# Combine all files into one DataFrame.
prices_df = pd.concat(df_list, ignore_index=True)
print("Combined Prices Data:")
print(prices_df.head())

# Convert price columns to numeric.
prices_df['mid_price'] = pd.to_numeric(prices_df['mid_price'], errors='coerce')
prices_df['bid_price_1'] = pd.to_numeric(prices_df['bid_price_1'], errors='coerce')
prices_df['ask_price_1'] = pd.to_numeric(prices_df['ask_price_1'], errors='coerce')
prices_df['timestamp'] = pd.to_numeric(prices_df['timestamp'], errors='coerce')
prices_df.dropna(subset=['mid_price'], inplace=True)

# --- Step 2: Summary Statistics ---
print("Summary Statistics for Prices Data:")
print(prices_df.describe())

# --- Step 3: EDA by Product ---

# Filter data for a specific product, e.g. SQUID_INK.
squid_df = prices_df[prices_df['product'] == "SQUID_INK"]
resin_df = prices_df[prices_df['product'] == "RAINFOREST_RESIN"]

# Plot mid_price over time for SQUID_INK.
plt.figure(figsize=(10, 5))
plt.plot(squid_df['timestamp'], squid_df['mid_price'], label="SQUID_INK Mid Price", color='blue')
plt.title("SQUID_INK Mid Price Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.legend()
plt.show()

# Plot distribution of mid_price for SQUID_INK.
plt.figure(figsize=(8, 5))
sns.histplot(squid_df['mid_price'], bins=30, kde=True, color='blue')
plt.title("Distribution of SQUID_INK Mid Price")
plt.xlabel("Mid Price")
plt.show()

# Calculate and plot the bid-ask spread.
prices_df['spread'] = prices_df['ask_price_1'] - prices_df['bid_price_1']
plt.figure(figsize=(8, 5))
sns.histplot(prices_df['spread'].dropna(), bins=30, kde=True, color='green')
plt.title("Distribution of Bid-Ask Spread")
plt.xlabel("Spread")
plt.show()

# --- Step 4: Load and Analyze Trade Data ---

# List your trade CSV files here.
trade_files = [
    "../data/trades_round_1_day_-1.csv",
    "../data/trades_round_1_day_-2.csv",
    "../data/trades_round_1_day_0.csv"
]

trade_dfs = []
for file in trade_files:
    trade_df = pd.read_csv(file, delimiter=';')
    trade_dfs.append(trade_df)

# Combine trade data.
trades_df = pd.concat(trade_dfs, ignore_index=True)
print("Combined Trades Data:")
print(trades_df.head())

# Convert price and quantity to numeric.
trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
trades_df['quantity'] = pd.to_numeric(trades_df['quantity'], errors='coerce')

# Filter trades for SQUID_INK (if the symbol column indicates the product).
squid_trades = trades_df[trades_df['symbol'] == "SQUID_INK"]

# Plot trade prices.
plt.figure(figsize=(10,5))
plt.plot(squid_trades['timestamp'], squid_trades['price'], 'o', markersize=3, label="Trade Price")
plt.title("SQUID_INK Trade Prices Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.legend()
plt.show()
