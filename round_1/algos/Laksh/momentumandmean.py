from datamodel import Order, TradingState, OrderDepth, UserId
from typing import List
import numpy as np
import jsonpickle

# ===== Helper Functions =====
def compute_ema(prices, alpha=None):
    """Compute exponential moving average (EMA) on a 1D numpy array.
       If alpha is not provided, use 2/(N+1) where N is the number of observations."""
    if alpha is None:
        alpha = 2.0 / (len(prices) + 1.0)
    ema = prices[0]
    for p in prices[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema

# ===== TRADER CLASS =====
class Trader:
    def run(self, state: TradingState):
        # Debug prints for clarity.
        print("traderData:", state.traderData)
        print("Observations:", str(state.observations))
        
        # Load persistent trader data.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}
        
        # For SQUID_INK, keep a persistent list of the last 10 mid prices.
        if "squid_price_history" not in trader_state:
            trader_state["squid_price_history"] = []
        price_history = trader_state["squid_price_history"]
        
        result = {}  # To hold orders for each product.
        
        # Process each product in the order book.
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # We will only trade SQUID_INK.
            if product != "SQUID_INK":
                result[product] = []
                continue
            
            # Determine current mid price from the order book.
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                current_mid = (best_bid + best_ask) / 2.0
            else:
                # Fallback if no order book information.
                current_mid = price_history[-1] if price_history else 1971.0
            print("SQUID_INK - Current mid price:", current_mid)
            
            # Append current price to history.
            price_history.append(current_mid)
            if len(price_history) > 10:
                price_history = price_history[-10:]
            trader_state["squid_price_history"] = price_history
            
            # Only perform our signal calculations if we have at least 10 prices.
            if len(price_history) < 10:
                print("SQUID_INK - Not enough history (need 10 prices). No trade executed.")
            else:
                # Convert history to numpy array.
                prices_np = np.array(price_history, dtype=np.float32)
                
                # Compute EMA using the custom function.
                ema = compute_ema(prices_np)
                # Compute volatility (standard deviation) from the last 10 prices.
                volatility = np.std(prices_np)
                # Compute momentum as the difference between current and previous price.
                momentum = current_mid - price_history[-2] if len(price_history) >= 2 else 0.0
                
                print("SQUID_INK - EMA:", ema, "Volatility:", volatility, "Momentum:", momentum)
                
                # Define adaptive thresholds (adjust multipliers as needed).
                threshold_mean = volatility * 1.0   # For mean reversion.
                threshold_momentum = volatility * 0.5 # For momentum.
                
                # Mean reversion signal: if price is above EMA by threshold, expect reversion downward.
                mean_signal = 0
                if current_mid > ema + threshold_mean:
                    mean_signal = -1
                elif current_mid < ema - threshold_mean:
                    mean_signal = 1
                
                # Momentum signal: if momentum exceeds threshold, signal the trend.
                momentum_signal = 0
                if momentum > threshold_momentum:
                    momentum_signal = 1
                elif momentum < -threshold_momentum:
                    momentum_signal = -1
                
                print("SQUID_INK - Mean signal:", mean_signal, "Momentum signal:", momentum_signal)
                
                # Combine signals: trade only if both signals agree.
                combined_signal = 0
                if mean_signal == momentum_signal and mean_signal != 0:
                    combined_signal = mean_signal
                print("SQUID_INK - Combined signal:", combined_signal)
                
                # Execute trade based on combined signal.
                if combined_signal == 1:
                    # BUY signal.
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        print("SQUID_INK - BUY order: Buying 1 unit at", best_ask)
                        orders.append(Order(product, best_ask, 1))
                    else:
                        print("SQUID_INK - BUY signal but no sell orders available.")
                elif combined_signal == -1:
                    # SELL signal.
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        print("SQUID_INK - SELL order: Selling 1 unit at", best_bid)
                        orders.append(Order(product, best_bid, -1))
                    else:
                        print("SQUID_INK - SELL signal but no buy orders available.")
                else:
                    print("SQUID_INK - No clear signal. No trade executed.")
            
            result[product] = orders
        
        # Store updated persistent state.
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion operations.
        return result, conversions, traderData
