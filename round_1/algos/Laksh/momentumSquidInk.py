from datamodel import Order, TradingState, OrderDepth, UserId
from typing import List
import numpy as np
import jsonpickle

class Trader:
    def run(self, state: TradingState):
        # Print initial traderData and observations for debugging.
        print("traderData:", state.traderData)
        print("Observations:", str(state.observations))
        
        # Load persistent state from previous iterations.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}
        
        # Initialize persistent storage for momentum trading on SQUID_INK.
        if "momentum_data" not in trader_state:
            # We'll store the last mid-price observed.
            trader_state["momentum_data"] = {"last_mid": None}
        momentum_data = trader_state["momentum_data"]
        
        result = {}  # Dictionary to collect orders per product.
        
        # Loop through available products in the order book.
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # # ---------------------------
            # # Strategy for RAINFOREST_RESIN: Fixed Acceptable Price Trading.
            # # ---------------------------
            # if product == "RAINFOREST_RESIN":
            #     acceptable_price = 10000  # Fixed target.
            #     print("RAINFOREST_RESIN - Acceptable Price:", acceptable_price)
            #     print("RAINFOREST_RESIN - Buy Order Depth:", len(order_depth.buy_orders),
            #           "Sell Order Depth:", len(order_depth.sell_orders))
            #     if len(order_depth.sell_orders) != 0:
            #         best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            #         if int(best_ask) < acceptable_price:
            #             print("RAINFOREST_RESIN - BUY order: Buying", -best_ask_amount, "units at", best_ask)
            #             orders.append(Order(product, best_ask, -best_ask_amount))
            #     if len(order_depth.buy_orders) != 0:
            #         best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            #         if int(best_bid) > acceptable_price:
            #             print("RAINFOREST_RESIN - SELL order: Selling", best_bid_amount, "units at", best_bid)
            #             orders.append(Order(product, best_bid, -best_bid_amount))
            #     result[product] = orders
            
            # ---------------------------
            # Strategy for SQUID_INK: Basic Momentum Trading.
            # ---------------------------
            if product == "SQUID_INK":
                # Determine the current mid-price from the order book.
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    current_mid = (best_bid + best_ask) / 2
                else:
                    # Fallback to a default value if order book information is incomplete.
                    current_mid = 1971.0
                print("SQUID_INK - Current mid price:", current_mid)
                
                # Retrieve the last stored mid-price.
                last_mid = momentum_data.get("last_mid")
                threshold = 9 # Adjust threshold based on backtesting.
                
                if last_mid is None:
                    print("SQUID_INK - No previous mid-price available; initializing with", current_mid)
                    # Initialize with the current mid-price; no trade on first data point.
                    momentum_data["last_mid"] = current_mid
                else:
                    # Compute momentum: current mid minus previous mid.
                    momentum = current_mid - last_mid
                    print("SQUID_INK - Previous mid price:", last_mid)
                    print("SQUID_INK - Momentum:", momentum)
                    
                    # Generate a trade signal based on the momentum value.
                    if momentum > threshold:
                        # Upward momentum – signal to buy.
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            print("SQUID_INK - Upward momentum detected. Placing BUY order for 1 unit at", best_ask)
                            orders.append(Order(product, best_ask, 1))
                        else:
                            print("SQUID_INK - Upward momentum signal but no sell orders available.")
                    elif momentum < -threshold:
                        # Downward momentum – signal to sell.
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            print("SQUID_INK - Downward momentum detected. Placing SELL order for 1 unit at", best_bid)
                            orders.append(Order(product, best_bid, -1))
                        else:
                            print("SQUID_INK - Downward momentum signal but no buy orders available.")
                    else:
                        print("SQUID_INK - Momentum change is insignificant. No trade executed.")
                    # Update the stored mid-price.
                    momentum_data["last_mid"] = current_mid
                result[product] = orders
                
            # For any other product, no orders are generated.
        
        # Encode and return the persistent state.
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion operations in this strategy.
        return result, conversions, traderData
