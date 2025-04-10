from datamodel import Order, TradingState, OrderDepth, UserId
from typing import List
import jsonpickle
import statistics

class Trader:
    def run(self, state: TradingState):
        # Load persistent trader state from previous iterations
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}

        # Initialize persistent data for SQUID_INK if not already present.
        if "squid_data" not in trader_state:
            trader_state["squid_data"] = {
                "last_mid": None,
                "rolling_prices": [],   # For use in the mean reversion strategy.
                "order_count": 0,
                "momentum_buy_orders": 0,
                "momentum_sell_orders": 0,
                "mean_rev_buy_orders": 0,
                "mean_rev_sell_orders": 0,
                "debug_logs": []
            }
        # Set the strategy to use if not yet specified.
        # Change this value to "mean_reversion" to test that strategy.
        if "strategy" not in trader_state["squid_data"]:
            trader_state["squid_data"]["strategy"] = "momentum"
        
        strategy = trader_state["squid_data"]["strategy"]
        print("SQUID_INK - Selected Strategy:", strategy)

        result = {}  # Dictionary to hold orders for each product

        # Loop only over SQUID_INK (ignore other products)
        for product in state.order_depths:
            if product != "SQUID_INK":
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Compute the best bid, best ask, and mid-price.
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                print("SQUID_INK - Best Bid:", best_bid, 
                      "Best Ask:", best_ask, 
                      "Mid Price:", mid_price)
            else:
                mid_price = 100  # Fallback mid-price if order book incomplete.
                print("SQUID_INK - Incomplete order book; using fallback mid price:", mid_price)
            
            # Update the rolling window for mean reversion strategy.
            rolling_prices = trader_state["squid_data"]["rolling_prices"]
            rolling_prices.append(mid_price)
            if len(rolling_prices) > 10:
                rolling_prices = rolling_prices[-10:]
            trader_state["squid_data"]["rolling_prices"] = rolling_prices

            # Decide which strategy to use.
            if strategy == "momentum":
                # ---------- Momentum Trading Strategy ----------
                last_mid = trader_state["squid_data"]["last_mid"]
                if last_mid is None:
                    print("SQUID_INK (Momentum) - No previous mid price, initializing to", mid_price)
                    trader_state["squid_data"]["last_mid"] = mid_price
                else:
                    momentum = mid_price - last_mid
                    momentum_pct = (momentum / last_mid * 100) if last_mid != 0 else 0
                    print("SQUID_INK (Momentum) - Last Mid:", last_mid, 
                          "Current Mid:", mid_price, 
                          "Momentum:", momentum, "(", round(momentum_pct, 2), "%)")
                    
                    threshold = 1.0  # Adjust the momentum threshold as needed.
                    current_position = state.position.get(product, 0)
                    print("SQUID_INK (Momentum) - Current Position:", current_position)
                    
                    if momentum > threshold:
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            order_qty = 1
                            print("SQUID_INK (Momentum) - Upward momentum detected; placing BUY order for", order_qty, "at", best_ask)
                            orders.append(Order(product, best_ask, order_qty))
                            trader_state["squid_data"]["momentum_buy_orders"] += 1
                        else:
                            print("SQUID_INK (Momentum) - Upward momentum but no sell orders available.")
                    elif momentum < -threshold:
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            order_qty = 1
                            print("SQUID_INK (Momentum) - Downward momentum detected; placing SELL order for", order_qty, "at", best_bid)
                            orders.append(Order(product, best_bid, -order_qty))
                            trader_state["squid_data"]["momentum_sell_orders"] += 1
                        else:
                            print("SQUID_INK (Momentum) - Downward momentum but no buy orders available.")
                    else:
                        print("SQUID_INK (Momentum) - Momentum below threshold (", threshold, "); no trade executed.")

                    trader_state["squid_data"]["last_mid"] = mid_price

            elif strategy == "mean_reversion":
                # ---------- Mean Reversion Strategy ----------
                if len(rolling_prices) >= 3:
                    avg_price = sum(rolling_prices) / len(rolling_prices)
                    deviation = mid_price - avg_price
                    print("SQUID_INK (Mean Reversion) - Rolling Average:", avg_price, 
                          "Current Mid:", mid_price, 
                          "Deviation:", deviation)
                    threshold = 0.5  # Adjust the deviation threshold as needed.
                    current_position = state.position.get(product, 0)
                    print("SQUID_INK (Mean Reversion) - Current Position:", current_position)
                    
                    if deviation > threshold:
                        # Price is significantly above average, take a contrarian SELL.
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            order_qty = 1
                            print("SQUID_INK (Mean Reversion) - Price above average; placing SELL order for", order_qty, "at", best_bid)
                            orders.append(Order(product, best_bid, -order_qty))
                            trader_state["squid_data"]["mean_rev_sell_orders"] += 1
                        else:
                            print("SQUID_INK (Mean Reversion) - Price above average but no buy orders available.")
                    elif deviation < -threshold:
                        # Price is significantly below average, take a contrarian BUY.
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            order_qty = 1
                            print("SQUID_INK (Mean Reversion) - Price below average; placing BUY order for", order_qty, "at", best_ask)
                            orders.append(Order(product, best_ask, order_qty))
                            trader_state["squid_data"]["mean_rev_buy_orders"] += 1
                        else:
                            print("SQUID_INK (Mean Reversion) - Price below average but no sell orders available.")
                    else:
                        print("SQUID_INK (Mean Reversion) - Deviation within threshold (", threshold, "); no trade executed.")
                else:
                    print("SQUID_INK (Mean Reversion) - Not enough data in rolling window; no trade executed.")
            else:
                print("SQUID_INK - Unknown strategy selected:", strategy)

            # Log debug information for this iteration.
            trader_state["squid_data"]["order_count"] += len(orders)
            debug_entry = {
                "timestamp": state.timestamp,
                "mid_price": mid_price,
                "rolling_prices": rolling_prices.copy(),
                "orders_placed": len(orders),
                "current_position": state.position.get(product, 0)
            }
            trader_state["squid_data"]["debug_logs"].append(debug_entry)
            print("SQUID_INK - Debug Log:", debug_entry)
            
            result[product] = orders

        # Encode the updated persistent state into traderData for the next iteration.
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion operations here.
        return result, conversions, traderData
