from datamodel import Order, TradingState, OrderDepth, UserId
from typing import List
import jsonpickle

class Trader:
    
    def run(self, state: TradingState):
        # Print the incoming persistent state and observations for debugging.
        print("traderData:", state.traderData)
        print("Observations:", state.observations)
        
        # Load persistent trader state (if any) from previous iterations.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}

        # Initialize persistent data for Rainforest Resin if not present.
        if "resin_data" not in trader_state:
            trader_state["resin_data"] = {
                "order_count": 0,
                "buy_orders": 0,
                "sell_orders": 0,
                "debug_logs": []
            }
        
        # Initialize persistent data for SQUID_INK if not present.
        if "squid_data" not in trader_state:
            trader_state["squid_data"] = {
                "last_mid": None,
                "order_count": 0,
                "buy_orders": 0,
                "sell_orders": 0,
                "debug_logs": []
            }
        
        result = {}  # Dictionary to hold orders for each product

        # Loop through all products in the order book.
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # --------------------------------------------------
            # Strategy for Rainforest Resin (Market Making)
            # --------------------------------------------------
            if product == "RAINFOREST_RESIN":
                # Fixed acceptable price as determined by your analysis.
                acceptable_price = 10000  
                print("RAINFOREST_RESIN - Acceptable Price:", acceptable_price)
                print("RAINFOREST_RESIN - Buy Order Depth (count):", len(order_depth.buy_orders),
                      "Sell Order Depth (count):", len(order_depth.sell_orders))
                
                # Process SELL orders: If a sell order's price is below acceptable, buy it.
                if len(order_depth.sell_orders) != 0:
                    # We take the first price level in the dictionary order (which may be sorted; in the simulation
                    # it is assumed that best quotes come first).
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < acceptable_price:
                        print("RAINFOREST_RESIN - Placing BUY order for", -best_ask_amount,
                              "unit(s) at price", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
                        trader_state["resin_data"]["buy_orders"] += 1
                    else:
                        print("RAINFOREST_RESIN - Best ask", best_ask, ">= acceptable price; no BUY order.")
                
                # Process BUY orders: If a buy order's price is above acceptable, sell it.
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > acceptable_price:
                        print("RAINFOREST_RESIN - Placing SELL order for", best_bid_amount,
                              "unit(s) at price", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))
                        trader_state["resin_data"]["sell_orders"] += 1
                    else:
                        print("RAINFOREST_RESIN - Best bid", best_bid, "<= acceptable price; no SELL order.")
                
                trader_state["resin_data"]["order_count"] += len(orders)
                # Log debug info for Rainforest Resin.
                resin_debug = {
                    "timestamp": state.timestamp,
                    "acceptable_price": acceptable_price,
                    "orders_placed": len(orders)
                }
                trader_state["resin_data"]["debug_logs"].append(resin_debug)
                print("RAINFOREST_RESIN - Debug Log:", resin_debug)
                # Save orders for Rainforest Resin.
                result[product] = orders

            # --------------------------------------------------
            # Strategy for SQUID_INK (Momentum Trading)
            # --------------------------------------------------
            elif product == "SQUID_INK":
                # Compute the best bid, best ask, and mid-price if available.
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    print("SQUID_INK - Best Bid:", best_bid, "Best Ask:", best_ask, "Mid Price:", mid_price)
                else:
                    mid_price = 100  # Fallback mid-price
                    print("SQUID_INK - Incomplete order book; using fallback mid price:", mid_price)
                
                # Retrieve last mid price from persistent state.
                last_mid = trader_state["squid_data"]["last_mid"]
                if last_mid is None:
                    print("SQUID_INK - Initializing last_mid to", mid_price, "; no trade this iteration.")
                    trader_state["squid_data"]["last_mid"] = mid_price
                else:
                    momentum = mid_price - last_mid
                    momentum_pct = (momentum / last_mid * 100) if last_mid != 0 else 0
                    print("SQUID_INK - Last Mid:", last_mid, "Current Mid:", mid_price)
                    print("SQUID_INK - Momentum:", momentum, "(", round(momentum_pct, 2), "% )")
                    
                    # Define threshold for momentum to take action.
                    threshold = 1.0  # You can adjust this value based on backtesting.
                    
                    # Get current position for risk management.
                    current_position = state.position.get(product, 0)
                    print("SQUID_INK - Current Position:", current_position)
                    
                    # If upward momentum is strong, buy at the best ask.
                    if momentum > threshold:
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            order_qty = 1  # Example order size; adjust as desired.
                            print("SQUID_INK - Upward momentum detected. Placing BUY order for", order_qty,
                                  "unit(s) at price", best_ask)
                            orders.append(Order(product, best_ask, order_qty))
                            trader_state["squid_data"]["buy_orders"] += 1
                        else:
                            print("SQUID_INK - Upward momentum but no sell orders; skipping BUY order.")
                    # If downward momentum is strong, sell at the best bid.
                    elif momentum < -threshold:
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            order_qty = 1  # Example order size.
                            print("SQUID_INK - Downward momentum detected. Placing SELL order for", order_qty,
                                  "unit(s) at price", best_bid)
                            orders.append(Order(product, best_bid, -order_qty))
                            trader_state["squid_data"]["sell_orders"] += 1
                        else:
                            print("SQUID_INK - Downward momentum but no buy orders; skipping SELL order.")
                    else:
                        print("SQUID_INK - Momentum change below threshold (", threshold, "). No trade executed.")
                    
                    # Update last_mid for next iteration.
                    trader_state["squid_data"]["last_mid"] = mid_price

                    # Log debug info for SQUID_INK.
                    squid_debug = {
                        "timestamp": state.timestamp,
                        "mid_price": mid_price,
                        "last_mid": last_mid,
                        "momentum": momentum,
                        "momentum_pct": round(momentum_pct, 2),
                        "orders_placed": len(orders),
                        "current_position": current_position
                    }
                    trader_state["squid_data"]["debug_logs"].append(squid_debug)
                    print("SQUID_INK - Debug Log:", squid_debug)
                
                # Save orders for SQUID_INK.
                result[product] = orders
        
        # Encode the updated persistent state into traderData.
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion operations in this algorithm.
        return result, conversions, traderData
