import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Data Loading and Preprocessing
# -------------------------------
price_files = [
    "../data/prices_round_1_day_-1.csv",
    "../data/prices_round_1_day_-2.csv",
    "../data/prices_round_1_day_0.csv"
]

df_list = []
for file in price_files:
    # Read the CSV file (adjust delimiter if necessary)
    df = pd.read_csv(file, delimiter=';')
    # Filter for SQUID_INK product rows.
    df = df[df['product'] == "SQUID_INK"]
    df_list.append(df)

# Combine the DataFrames and sort by timestamp.
data = pd.concat(df_list, ignore_index=True)
data = data.sort_values(by="timestamp")

# Convert mid_price to numeric and drop any rows with missing values.
data['mid_price'] = pd.to_numeric(data['mid_price'], errors='coerce')
data = data.dropna(subset=['mid_price'])
print("Total SQUID_INK mid_price data points:", len(data))

# -------------------------------
# Step 2: Construct Lagged Features
# -------------------------------
# For example, using a lag window of 6: each sample uses the previous 6 mid_prices 
# as features and the current mid_price as the target.
lag = 6
X_list = []
y_list = []
mid_prices = data['mid_price'].values

for i in range(lag, len(mid_prices)):
    X_list.append(mid_prices[i - lag : i])  # previous 6 mid_prices
    y_list.append(mid_prices[i])           # current mid_price as target

X = np.array(X_list)
y = np.array(y_list)
print("Shape of feature matrix X:", X.shape)
print("Shape of target vector y:", y.shape)

# -------------------------------
# Step 3: Split Data into Training and Testing Sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])

# -------------------------------
# Step 4: Train the XGBoost Model
# -------------------------------
model = xgb.XGBRegressor(
    n_estimators=1000, 
    max_depth=5, 
    learning_rate=0.01, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate the Model
# -------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)

# -------------------------------
# Step 6: Feature Importance Plot (Optional)
# -------------------------------
xgb.plot_importance(model)
plt.title("XGBoost Feature Importance for SQUID_INK")
plt.xlabel("Importance Score")
plt.ylabel("Feature Index (Lag order)")
plt.show()
