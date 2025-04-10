from datamodel import Order, TradingState, OrderDepth
from typing import List
import numpy as np
import jsonpickle

def compute_ema(prices, span):
    """Compute an exponential moving average for a 1D numpy array using pandas' ewm (as a simpler implementation)."""
    # Using a simple numpy-based implementation as well.
    alpha = 2.0 / (span + 1.0)
    ema = prices[0]
    for p in prices[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema

class Trader:
    def run(self, state: TradingState):
        # Print initial debugging information.
        print("traderData:", state.traderData)
        print("Observations:", str(state.observations))
        
        # Load persistent state.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}
        
        # Persistent price history for SQUID_INK.
        if "squid_price_history" not in trader_state:
            trader_state["squid_price_history"] = []
        price_history = trader_state["squid_price_history"]
        
        result = {}  # To hold orders for each product.

        # Process each product—this script will trade only SQUID_INK.
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product != "SQUID_INK":
                result[product] = []
                continue
            
            # Determine current mid price.
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                current_mid = (best_bid + best_ask) / 2.0
            else:
                # If order book data is missing, use last known price or a fallback.
                current_mid = price_history[-1] if price_history else 1971.0
            print("SQUID_INK - Current mid price:", current_mid)
            
            # Append current mid price to persistent history.
            price_history.append(current_mid)
            # Use a window of the most recent 20 prices (adjust as you wish).
            window = 53
            if len(price_history) > window:
                price_history = price_history[-window:]
            trader_state["squid_price_history"] = price_history
            
            # Only proceed if we have enough data.
            if len(price_history) < window:
                print("SQUID_INK - Not enough history for analysis (need at least {} data points).".format(window))
            else:
                prices = np.array(price_history, dtype=np.float32)
                # Use EMA to get a weighted moving average.
                ema = compute_ema(prices, span=window)
                # Rolling volatility: standard deviation of the window.
                vol = np.std(prices)
                # Compute deviation (z-score).
                if vol > 0:
                    z_score = (current_mid - ema) / vol
                else:
                    z_score = 0.0
                
                print("SQUID_INK - EMA:", ema, "Volatility:", vol, "Z-score:", z_score)
                
                # Define an adaptive threshold.
                # For example, if we set the threshold to be 1.0, but we can adjust by volatility if needed.
                threshold = 1.77  # You can adjust or even make this a function of volatility.
                
                # Generate a mean reversion signal:
                # If z_score is very high, price might revert downward.
                # If z_score is very low, price might revert upward.
                signal = 0
                if z_score > threshold:
                    signal = -1   # SELL signal.
                elif z_score < -threshold:
                    signal = 1    # BUY signal.
                
                print("SQUID_INK - Signal:", signal)
                
                # Execute trade based on signal.
                if signal == 1:
                    # Attempt to BUY if there is an ask price.
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        print("SQUID_INK - BUY order: Buying 1 unit at", best_ask)
                        orders.append(Order(product, best_ask, 1))
                    else:
                        print("SQUID_INK - BUY signal, but no sell orders available.")
                elif signal == -1:
                    # Attempt to SELL if there is a bid price.
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        print("SQUID_INK - SELL order: Selling 1 unit at", best_bid)
                        orders.append(Order(product, best_bid, -1))
                    else:
                        print("SQUID_INK - SELL signal, but no buy orders available.")
                else:
                    print("SQUID_INK - No significant deviation signal. No trade executed.")
            
            result[product] = orders
        
        # Encode the updated persistent state.
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion operations.
        return result, conversions, traderData
