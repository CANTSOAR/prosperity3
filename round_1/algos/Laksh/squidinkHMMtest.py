from datamodel import Order, TradingState, OrderDepth, UserId
from typing import List
import numpy as np
import math
import jsonpickle

# ============================================================
# HMM PARAMETERS (Hard-code these offline-trained values)
# Replace placeholder values with your actual parameters.
# Assume a 3-state HMM.
#   π: initial state probabilities, shape (3,)
#   A: state transition matrix, shape (3, 3)
#   means: emission means for each state, shape (3,)
#   stds: emission standard deviations for each state, shape (3,)
# ============================================================
pi = np.array([0.000000,
  0.000000,
  1.000000], dtype=np.float32)  # Example values; replace with your trained π.
A = np.array([[0.036166, 0.963834, 0.000000],
  [0.407486, 0.592514, 0.000000],
  [0.000069, 0.000000, 0.999931]], dtype=np.float32)   # Example values; replace with your trained A.
means = np.array([
1873.943683,
  2020.548436,
  1964.852742
], dtype=np.float32)  # Replace with your emission means.
stds = np.array([
  31.028891,
  71.553839,
  22.764139
], dtype=np.float32)         # Replace with your emission stds.

# ============================================================
# Gaussian PDF function.
# ============================================================
def gaussian_pdf(x, mean, std):
    return (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std) ** 2)

# ============================================================
# Forward Algorithm for the HMM.
# Given a sequence of observations (obs_seq), compute the forward probabilities.
# ============================================================
def forward_algorithm(obs_seq):
    n_states = len(pi)
    T = len(obs_seq)
    alpha = np.zeros((T, n_states), dtype=np.float32)
    # Initialization for t = 0.
    for i in range(n_states):
        alpha[0, i] = pi[i] * gaussian_pdf(obs_seq[0], means[i], stds[i])
    # Recursion: compute alpha[t, j] = (sum_i alpha[t-1,i] * A[i, j]) * p(obs[t]|state j)
    for t in range(1, T):
        for j in range(n_states):
            sum_prev = 0.0
            for i in range(n_states):
                sum_prev += alpha[t-1, i] * A[i, j]
            alpha[t, j] = sum_prev * gaussian_pdf(obs_seq[t], means[j], stds[j])
    return alpha  # shape (T, n_states)

# ============================================================
# TRADER CLASS: Combined Strategy with HMM-Based Regime Detection for SQUID_INK
# and Fixed Acceptable Price Strategy for RAINFOREST_RESIN.
# ============================================================
class Trader:
    def run(self, state: TradingState):
        # Print initial traderData and observations (for debugging).
        print("traderData:", state.traderData)
        print("Observations:", str(state.observations))
        
        # Load persistent state.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}
        
        # Initialize persistent storage for SQUID_INK HMM.
        if "squid_hmm" not in trader_state:
            # We'll store a rolling window of mid-price observations.
            trader_state["squid_hmm"] = {"price_history": []}
        hmm_state = trader_state["squid_hmm"]
        
        result = {}
        
        # Loop through products.
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            
            # ----- Strategy for RAINFOREST_RESIN: Fixed Price Trading -----
            if product == "RAINFOREST_RESIN":
                orders: List[Order] = []
                acceptable_price = 10000  # Fixed acceptable price.
                print("RAINFOREST_RESIN - Acceptable Price:", acceptable_price)
                print("RAINFOREST_RESIN - Buy Order Depth:", len(order_depth.buy_orders),
                      "Sell Order Depth:", len(order_depth.sell_orders))
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < acceptable_price:
                        print("RAINFOREST_RESIN - BUY order: Buying", -best_ask_amount, "units at", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > acceptable_price:
                        print("RAINFOREST_RESIN - SELL order: Selling", best_bid_amount, "units at", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))
                result[product] = orders
            
            # ----- Strategy for SQUID_INK: HMM-Regime Based Trading -----
            elif product == "SQUID_INK":
                orders: List[Order] = []
                # Determine the current mid-price from the order book.
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    current_mid = (best_bid + best_ask) / 2
                else:
                    # Fallback to average of the HMM means.
                    current_mid = np.mean(means)
                print("SQUID_INK - Current mid price:", current_mid)
                
                # Update rolling price history (store recent observations; using window length 5 here).
                hmm_state["price_history"].append(current_mid)
                if len(hmm_state["price_history"]) > 5:
                    hmm_state["price_history"] = hmm_state["price_history"][-5:]
                
                if len(hmm_state["price_history"]) < 5:
                    print("SQUID_INK - Not enough history for HMM prediction. Skipping trade.")
                else:
                    # Use the most recent 5 observations for the forward algorithm.
                    obs_seq = np.array(hmm_state["price_history"], dtype=np.float32)
                    alpha = forward_algorithm(obs_seq)
                    final_alpha = alpha[-1]  # probabilities for final time step.
                    total = np.sum(final_alpha)
                    if total > 0:
                        state_probs = final_alpha / total
                    else:
                        state_probs = final_alpha
                    most_likely_state = np.argmax(state_probs)
                    print("SQUID_INK - HMM state probabilities:", state_probs)
                    print("SQUID_INK - Most likely hidden state:", most_likely_state)
                    
                    # Example trading logic: assume state 2 is bullish, state 0 is bearish.
                    bullish_state = 2
                    bearish_state = 0
                    prob_diff = state_probs[bullish_state] - state_probs[bearish_state]
                    threshold = 0.1  # Require a minimum difference.
                    
                    if most_likely_state == bullish_state and prob_diff > threshold:
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            print("SQUID_INK - Bullish regime detected. Placing BUY order for 1 unit at", best_ask)
                            orders.append(Order(product, best_ask, 1))
                        else:
                            print("SQUID_INK - Bullish regime but no sell orders available.")
                    elif most_likely_state == bearish_state and (-prob_diff) > threshold:
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            print("SQUID_INK - Bearish regime detected. Placing SELL order for 1 unit at", best_bid)
                            orders.append(Order(product, best_bid, -1))
                        else:
                            print("SQUID_INK - Bearish regime but no buy orders available.")
                    else:
                        print("SQUID_INK - No clear regime signal. No trade executed.")
                result[product] = orders
        
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion operations.
        return result, conversions, traderData
