import pandas as pd

# Load one of your price CSV files and filter for SQUID_INK
df = pd.read_csv("../data/prices_round_1_day_-1.csv", delimiter=';')
df = df[df['product'] == "SQUID_INK"]
df['mid_price'] = pd.to_numeric(df['mid_price'], errors='coerce')
print(df['mid_price'].describe())
