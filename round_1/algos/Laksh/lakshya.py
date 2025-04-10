from datamodel import Order, TradingState, OrderDepth
from typing import List
import jsonpickle

class Trader:
    def run(self, state: TradingState):
        """
        This run() method is called every iteration with the latest market state.
        We use traderData (a string) to maintain persistent state (like historical prices)
        between iterations. If no traderData is available, we initialize it as an empty dictionary.
        """
        # Deserialize persistent state from traderData
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}


        result = {}  # Dictionary to hold orders for each product
        print(state.market_trades)
        # Iterate over every product with available market quotes
        for product, order_depth in state.order_depths.items():
            orders = []  # List to hold orders for this product
            current_position = state.position.get(product, 0)
            position_limit = 50  # Given per-product limit


            # Helper functions to calculate how much we can trade without breaching limits.
            def get_buy_capacity(pos):
                return position_limit - pos if pos >= 0 else position_limit + pos

            def get_sell_capacity(pos):
                return position_limit + pos

            # Calculate the mid price if both sides of the order book are available.
            mid_price = None
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2

            # =============================
            # Strategy 1: Rainforest Resin – Market Making
            # =============================
            if product == "RAINFOREST_RESIN":
                acceptable_price = 10000
                print("Acceptable price : " + str(acceptable_price))
                print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
        
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < acceptable_price:
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
        
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > acceptable_price:
                        print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))

            # =============================
            # Strategy 2: Squid Ink – Trend-Following with Stop-Loss
            # =============================
            elif product == "SQUID_INK":
                if mid_price is not None:
                    last_mid = trader_state.get("SQUID_INK_last_mid", mid_price)
                    threshold = 0.5
                    if mid_price > last_mid + threshold:
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            available_qty = get_buy_capacity(current_position)
                            order_qty = min(5, available_qty)
                            if order_qty > 0:
                                orders.append(Order(product, best_ask, order_qty))
                    elif mid_price < last_mid - threshold:
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            available_qty = get_sell_capacity(current_position)
                            order_qty = min(5, available_qty)
                            if order_qty > 0:
                                orders.append(Order(product, best_bid, -order_qty))
                    trader_state["SQUID_INK_last_mid"] = mid_price

            # =============================
            # Strategy 3: Kelp – Mean Reversion (Fade Strategy)
            # =============================
            elif product == "KELP":
                if mid_price is not None:
                    prev_ma = trader_state.get("KELP_ma", mid_price)
                    alpha = 0.2
                    new_ma = alpha * mid_price + (1 - alpha) * prev_ma
                    trader_state["KELP_ma"] = new_ma
                    threshold = 0.5
                    if mid_price > new_ma + threshold:
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            available_qty = get_sell_capacity(current_position)
                            order_qty = min(5, available_qty)
                            if order_qty > 0:
                                orders.append(Order(product, best_bid, -order_qty))
                    elif mid_price < new_ma - threshold:
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            available_qty = get_buy_capacity(current_position)
                            order_qty = min(5, available_qty)
                            if order_qty > 0:
                                orders.append(Order(product, best_ask, order_qty))

            if orders:
                result[product] = orders
    

        # Serialize the updated persistent state back into a string.
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion requests are used in this example.
        return result, conversions, traderData
