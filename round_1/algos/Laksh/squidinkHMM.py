import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

def train_hmm(file_paths, product="SQUID_INK", n_components=3):
    """
    Loads CSV files, filters for the given product, trains a Gaussian HMM on the mid_price data,
    and returns the trained model, the combined data, and the predicted hidden states.
    """
    df_list = []
    for file in file_paths:
        print("Loading file:", file)
        df = pd.read_csv(file, delimiter=';')
        # Filter for the specific product.
        df = df[df['product'] == product]
        df_list.append(df)
        
    data = pd.concat(df_list, ignore_index=True)
    data = data.sort_values(by="timestamp")
    # Convert mid_price to numeric and drop rows with missing values.
    data['mid_price'] = pd.to_numeric(data['mid_price'], errors='coerce')
    data.dropna(subset=['mid_price'], inplace=True)
    
    print(f"Total {product} mid_price data points: {len(data)}")
    
    # Extract the mid_price series and reshape to (-1, 1)
    prices = data['mid_price'].values.reshape(-1, 1)
    
    # Optionally, you might want to compute returns or differences – here we simply use prices.
    # Train a Gaussian HMM.
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=10000, random_state=42)
    model.fit(prices)
    
    # Predict the hidden states for the entire series.
    hidden_states = model.predict(prices)
    
    return model, data, hidden_states

def plot_hidden_states(data, hidden_states, product="SQUID_INK"):
    """
    Plots the mid_price over timestamp, color-coded by the hidden state.
    """
    plt.figure(figsize=(12, 6))
    for state in np.unique(hidden_states):
        idx = (hidden_states == state)
        plt.plot(data['timestamp'].values[idx], data['mid_price'].values[idx], '.', label=f"State {state}")
    plt.xlabel("Timestamp")
    plt.ylabel("Mid Price")
    plt.title(f"{product} Mid Price Time Series with HMM Hidden States")
    plt.legend()
    plt.show()

def print_model_parameters(model):
    """
    Prints the learned mean and covariance for each hidden state.
    """
    n_components = model.n_components
    for i in range(n_components):
        print(f"State {i}:")
        print("  Mean:", model.means_[i])
        print("  Covariance:", model.covars_[i])
        print("--------------------------------------------------")

def save_hmm_parameters(model, output_file="hmm_parameters.txt"):
    """
    Saves the HMM parameters (initial probabilities, transition matrix, means, and standard deviations)
    to a file in a format suitable for hard-coding.
    """
    with open(output_file, "w", encoding="utf-8") as f:  # Use UTF-8 encoding
        f.write("# ============================================================\n")
        f.write("# HMM PARAMETERS (Hard-code these offline-trained values)\n")
        f.write("# Assume a 3-state HMM.\n")
        f.write("#   pi: initial state probabilities, shape (3,)\n")  # Replace π with pi
        f.write("#   A: state transition matrix, shape (3, 3)\n")
        f.write("#   means: emission means for each state, shape (3,)\n")
        f.write("#   stds: emission standard deviations for each state, shape (3,)\n")
        f.write("# ============================================================\n\n")
        
        # Write initial state probabilities
        f.write("pi = [\n")
        f.write(",\n".join(f"  {p:.6f}" for p in model.startprob_))
        f.write("\n]\n\n")
        
        # Write state transition matrix
        f.write("A = [\n")
        for row in model.transmat_:
            f.write("  [" + ", ".join(f"{p:.6f}" for p in row) + "],\n")
        f.write("]\n\n")
        
        # Write means
        f.write("means = [\n")
        f.write(",\n".join(f"  {mean[0]:.6f}" for mean in model.means_))
        f.write("\n]\n\n")
        
        # Write standard deviations (square root of covariances)
        f.write("stds = [\n")
        f.write(",\n".join(f"  {np.sqrt(cov[0, 0]):.6f}" for cov in model.covars_))
        f.write("\n]\n")

if __name__ == "__main__":
    # List of CSV files to process.
    price_files = [
        "../data/prices_round_1_day_-1.csv",
        "../data/prices_round_1_day_-2.csv",
        "../data/prices_round_1_day_0.csv"
    ]
    
    # Train the HMM – you can adjust n_components based on your expectations.
    model, data, hidden_states = train_hmm(price_files, product="SQUID_INK", n_components=3)
    
    # Print out the learned parameters.
    print("\nTrained HMM Parameters:")
    print_model_parameters(model)
    
    # Plot the time series colored by the hidden state.
    plot_hidden_states(data, hidden_states, product="SQUID_INK")
    
    # Optional: Plot the histogram of state durations.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.hist(hidden_states, bins=np.arange(model.n_components+1)-0.5, edgecolor='black')
    plt.xlabel("Hidden State")
    plt.ylabel("Frequency")
    plt.title("Distribution of Hidden States")
    plt.show()
    
    # Save the trained HMM parameters to a file.
    save_hmm_parameters(model, output_file="hmm_parameters.txt")
