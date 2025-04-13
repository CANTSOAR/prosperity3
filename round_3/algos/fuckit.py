from typing import List
import string
import collections
import json
from typing import Any
import numpy as np
import pandas as pd
import math
from statistics import NormalDist

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:

    LIMITS = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
        "VOLCANIC_ROCK" : 400,
        "VOLCANIC_ROCK_VOUCHER_9500" : 200,
        "VOLCANIC_ROCK_VOUCHER_9750" : 200,
        "VOLCANIC_ROCK_VOUCHER_10000" : 200,
        "VOLCANIC_ROCK_VOUCHER_10250" : 200,
        "VOLCANIC_ROCK_VOUCHER_10500" : 200
    }

    BIDS = {
        "RAINFOREST_RESIN": [],
        "KELP": [],
        "SQUID_INK": [],
        "CROISSANTS": [],
        "JAMS": [],
        "DJEMBES": [],
        "PICNIC_BASKET1": [],
        "PICNIC_BASKET2": [],
        "VOLCANIC_ROCK" : [],
        "VOLCANIC_ROCK_VOUCHER_9500" : [],
        "VOLCANIC_ROCK_VOUCHER_9750" : [],
        "VOLCANIC_ROCK_VOUCHER_10000" : [],
        "VOLCANIC_ROCK_VOUCHER_10250" : [],
        "VOLCANIC_ROCK_VOUCHER_10500" : []
    }

    ASKS = {
        "RAINFOREST_RESIN": [],
        "KELP": [],
        "SQUID_INK": [],
        "CROISSANTS": [],
        "JAMS": [],
        "DJEMBES": [],
        "PICNIC_BASKET1": [],
        "PICNIC_BASKET2": [],
        "VOLCANIC_ROCK" : [],
        "VOLCANIC_ROCK_VOUCHER_9500" : [],
        "VOLCANIC_ROCK_VOUCHER_9750" : [],
        "VOLCANIC_ROCK_VOUCHER_10000" : [],
        "VOLCANIC_ROCK_VOUCHER_10250" : [],
        "VOLCANIC_ROCK_VOUCHER_10500" : []
    }

    kelp_last_bid = 2028
    kelp_last_ask = 2032

    ink_lock = False

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def populate_prices(self):
        for COMPONENT in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]:
            component_order_depth = self.state.order_depths[COMPONENT]

            ordered_sell_dict = collections.OrderedDict(sorted(component_order_depth.sell_orders.items()))
            ordered_buy_dict = collections.OrderedDict(sorted(component_order_depth.buy_orders.items(), reverse=True))

            best_ask = [ask for ask, _ in ordered_sell_dict.items()][0]
            best_bid = [bid for bid, _ in ordered_buy_dict.items()][0]

            self.ASKS[COMPONENT].append(best_ask)
            self.BIDS[COMPONENT].append(best_bid)
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        self.exit = False
        result = {}
        self.state = state
        self.POSITIONS = state.position

        COMPUTE_ORDERS = {
            "RAINFOREST_RESIN": self.compute_orders_resin,
            "SQUID_INK": self.compute_orders_ink,
            "KELP": self.compute_orders_kelp,
            "PICNIC_BASKET1": self.compute_orders_basket_1,
            "PICNIC_BASKET2": self.compute_orders_basket_2,
            "VOLCANIC_ROCK" : self.compute_orders_basket_1, # FILLER
            "VOLCANIC_ROCK_VOUCHER_9500" : self.compute_orders_options,
            "VOLCANIC_ROCK_VOUCHER_9750" : self.compute_orders_options,
            "VOLCANIC_ROCK_VOUCHER_10000" : self.compute_orders_options,
            "VOLCANIC_ROCK_VOUCHER_10250" : self.compute_orders_options,
            "VOLCANIC_ROCK_VOUCHER_10500" : self.compute_orders_options
        }

        self.populate_prices()

        for product in [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"   
        ]:
            order_depth: OrderDepth = state.order_depths[product]
            
            # For options products, extract strike price from name and provide volatility
            if "VOUCHER" in product:
                # Extract strike price from product name (e.g., "VOLCANIC_ROCK_VOUCHER_9500" -> 9500)
                strike_price = float(product.split("_")[-1])
                # Define a volatility value (you'll need to determine the appropriate value)
                volatility = 0.2  # Example value, adjust based on your strategy
                orders = COMPUTE_ORDERS[product](product, order_depth, volatility, strike_price)
            else:
                orders = COMPUTE_ORDERS[product](product, order_depth)
            
            result[product] = orders
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def compute_orders_resin(self, PRODUCT, order_depth):
        """
        LOGIC:

        if the current ask price is LOWER than the acceptable ask, then we snipe it, looking to later SELL at a higher price

        if the current bid price is HIGHER than the acceptable bid, then we snipe it, looking to later BUY at a lower price

        acceptable_bid: the LOWEST price that we are willing to SELL at
        acceptable_ask: the HIGHEST price that we are willing to BUY at
        """

        orders: list[Order] = []

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        best_remaining_ask = [ask for ask, _ in ordered_sell_dict.items()][-1]
        best_remaining_bid = [bid for bid, _ in ordered_buy_dict.items()][-1]

        acceptable_ask = 9999
        acceptable_bid = 10001

        undercut_amount = 1

        for ask, vol in ordered_sell_dict.items():
            if ask <= acceptable_ask and current_pos < self.LIMITS[PRODUCT]:
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos) # take the minimum of available volume and the volume we are allowed to take
                orders.append(Order(PRODUCT, ask, order_vol)) # this is a BUY order, we undercut by paying a little MORE
                current_pos += order_vol
            elif ask < best_remaining_ask:
                best_remaining_ask = ask

        for bid, vol in ordered_buy_dict.items():
            if bid >= acceptable_bid and current_pos > -self.LIMITS[PRODUCT]:
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos) # take the minimum of available volume and the volume we are allowed to take
                orders.append(Order(PRODUCT, bid, order_vol)) # this is a SELL order, we undercut by selling a bit CHEAPER
                current_pos += order_vol
            elif bid > best_remaining_bid:
                best_remaining_bid = bid

        if current_pos > -self.LIMITS[PRODUCT] and best_remaining_ask > acceptable_ask:
            order_vol = int((-self.LIMITS[PRODUCT] - current_pos) * .7)
            orders.append(Order(PRODUCT, best_remaining_ask - undercut_amount, order_vol))

        if current_pos < self.LIMITS[PRODUCT] and best_remaining_bid < acceptable_bid:
            order_vol = int((self.LIMITS[PRODUCT] - current_pos) * .7)
            orders.append(Order(PRODUCT, best_remaining_bid + undercut_amount, order_vol))

        return orders

    def compute_orders_kelp(self, PRODUCT, order_depth):
        """
        LOGIC:
        
        acceptable_bid and acceptable_ask are fake variables, dont actually do anything

        on each round, we recalculate a baseline value, by just seeing if we have two in a row

        then, set acceptable_bid to TWO above the baseline bid, and acceptable_ask to TWO below the baseline ask, not sure why, felt right

        then, logic is, SELL to any bid orders above the acceptable bid, because these are 'OVER VALUED'
        on flip side, BUY any ask orders below the acceptable ask, because these are 'UNDER VALUED'

        market taking ^^^

        then, make passive bids one above the baseline, 
        """
        orders: list[Order] = []

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        best_remaining_ask = [ask for ask, _ in ordered_sell_dict.items()][-1]
        best_remaining_bid = [bid for bid, _ in ordered_buy_dict.items()][-1]

        best_ask = [ask for ask, _ in ordered_sell_dict.items()][0]
        best_bid = [bid for bid, _ in ordered_buy_dict.items()][0]

        if best_ask == self.kelp_last_ask:
            self.ASKS[PRODUCT].append(best_ask)
        else:
            if self.ASKS[PRODUCT]:
                self.ASKS[PRODUCT].append(self.ASKS[PRODUCT][-1])
            else:
                self.ASKS[PRODUCT].append(self.kelp_last_ask)
            self.kelp_last_ask = best_ask

        if best_bid == self.kelp_last_bid:
            self.BIDS[PRODUCT].append(best_bid)
        else:
            if self.BIDS[PRODUCT]:
                self.BIDS[PRODUCT].append(self.BIDS[PRODUCT][-1])
            else:
                self.BIDS[PRODUCT].append(self.kelp_last_bid)
            self.kelp_last_bid = best_bid

        acceptable_ask = self.ASKS[PRODUCT][-1] - 2
        acceptable_bid = self.BIDS[PRODUCT][-1] + 2

        undercut_amount = 1

        for ask, vol in ordered_sell_dict.items():
            if ask <= acceptable_ask and current_pos < self.LIMITS[PRODUCT]:
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos) # take the minimum of available volume and the volume we are allowed to take
                orders.append(Order(PRODUCT, ask, order_vol)) # this is a BUY order, we undercut by paying a little MORE
                current_pos += order_vol
            elif ask < best_remaining_ask:
                best_remaining_ask = ask

        for bid, vol in ordered_buy_dict.items():
            if bid >= acceptable_bid and current_pos > -self.LIMITS[PRODUCT]:
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos) # take the minimum of available volume and the volume we are allowed to take
                orders.append(Order(PRODUCT, bid, order_vol)) # this is a SELL order, we undercut by selling a bit CHEAPER
                current_pos += order_vol
            elif bid > best_remaining_bid:
                best_remaining_bid = bid

        if current_pos > -self.LIMITS[PRODUCT] and best_remaining_ask > acceptable_ask:
            order_vol = int((-self.LIMITS[PRODUCT] - current_pos) * .7)
            orders.append(Order(PRODUCT, best_remaining_ask - undercut_amount, order_vol))

        if current_pos < self.LIMITS[PRODUCT] and best_remaining_bid < acceptable_bid:
            order_vol = int((self.LIMITS[PRODUCT] - current_pos) * .3)
            orders.append(Order(PRODUCT, best_remaining_bid + undercut_amount, order_vol))

        return orders
    
    def compute_orders_ink(self, PRODUCT, order_depth):
        orders: list[Order] = []

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        best_ask = [ask for ask, _ in ordered_sell_dict.items()][0]
        best_bid = [bid for bid, _ in ordered_buy_dict.items()][0]

        self.ASKS[PRODUCT].append(best_ask)
        self.BIDS[PRODUCT].append(best_bid)

        mid_prices = (pd.Series(self.ASKS[PRODUCT]) + pd.Series(self.BIDS[PRODUCT])) / 2
        lookback = 100

        if len(mid_prices) > lookback:
            moving_average = mid_prices.rolling(lookback).mean()
            standard_dev = mid_prices.rolling(lookback).std()

            if standard_dev.values[-1] < 1:
                return orders

            z_score = (mid_prices.values[-1] - moving_average.values[-1]) / standard_dev.values[-1]
            prev_z_score = (mid_prices.values[-2] - moving_average.values[-2]) / standard_dev.values[-2]

            z_score_thresh = 1.96

            logger.print(f"z-score: {z_score:.2f}, previous: {prev_z_score:.2f}")

            # LONG SIGNAL: high negative z-score, starting to revert upward
            long_entry = prev_z_score < -z_score_thresh and z_score > prev_z_score

            # SHORT SIGNAL: high positive z-score, starting to revert downward
            short_entry = prev_z_score > z_score_thresh and z_score < prev_z_score

            if long_entry and current_pos < self.LIMITS[PRODUCT]:
                for ask, vol in list(ordered_sell_dict.items())[:2]:
                    order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos)
                    orders.append(Order(PRODUCT, ask, order_vol))
                    current_pos += order_vol

            if short_entry and current_pos > -self.LIMITS[PRODUCT]:
                for bid, vol in list(ordered_buy_dict.items())[:2]:
                    order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos)
                    orders.append(Order(PRODUCT, bid, order_vol))
                    current_pos += order_vol

        return orders
    
    def compute_orders_basket_1(self, PRODUCT, order_depth):
        orders: list[Order] = []

        COMPONENTS = ["CROISSANTS", "JAMS","DJEMBES"]

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        lookback = 100

        mid_prices = (pd.Series(self.ASKS[PRODUCT]) + pd.Series(self.BIDS[PRODUCT])) / 2
        if len(mid_prices) < lookback:
            return orders

        component_mid_prices = np.array([(pd.Series(self.ASKS[COMPONENT]) + pd.Series(self.BIDS[COMPONENT])) / 2 for COMPONENT in COMPONENTS])
        estimated_mid_prices = component_mid_prices.T @ np.array([6, 1, 3])
        
        X = estimated_mid_prices[-lookback:].reshape(-1, 1)
        y = mid_prices[-lookback:]

        hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]

        estimated_mid_prices *= 1.0008

        bid_spread = pd.Series(self.BIDS[PRODUCT]).values - estimated_mid_prices
        ask_spread = pd.Series(self.ASKS[PRODUCT]).values - estimated_mid_prices

        bid_z_score = (bid_spread - bid_spread.mean()) / bid_spread.std()
        ask_z_score = (ask_spread - ask_spread.mean()) / ask_spread.std()

        z_score_reversal_threshold = 1.75

        long_reversal_entry = ask_z_score[-1] < -z_score_reversal_threshold and ask_z_score[-2] > -z_score_reversal_threshold
        short_reversal_entry = ask_z_score[-1] > z_score_reversal_threshold and ask_z_score[-2] < z_score_reversal_threshold

        exit = current_pos * ask_z_score[-1] > 0 or current_pos * bid_z_score[-1] > 0

        logger.print(bid_z_score[-1], ask_z_score[-1])
        logger.print(long_reversal_entry, short_reversal_entry)
        logger.print(exit, current_pos) 

        for ask, vol in list(ordered_sell_dict.items()):
            if (long_reversal_entry and current_pos < self.LIMITS[PRODUCT]) or (exit and current_pos < 0):
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, ask, order_vol))
                current_pos += order_vol

        for bid, vol in list(ordered_buy_dict.items()):
            if (short_reversal_entry and current_pos > -self.LIMITS[PRODUCT]) or (exit and current_pos > 0):
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, bid, order_vol))
                current_pos += order_vol

        return orders
   
    
    def compute_orders_basket_2(self, PRODUCT, order_depth):
        orders: list[Order] = []

        COMPONENTS = ["CROISSANTS", "JAMS"]

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        lookback = 100

        mid_prices = (pd.Series(self.ASKS[PRODUCT]) + pd.Series(self.BIDS[PRODUCT])) / 2
        if len(mid_prices) < lookback:
            return orders

        component_mid_prices = np.array([(pd.Series(self.ASKS[COMPONENT]) + pd.Series(self.BIDS[COMPONENT])) / 2 for COMPONENT in COMPONENTS])
        estimated_mid_prices = component_mid_prices.T @ np.array([4, 2])
        
        X = estimated_mid_prices[-lookback:].reshape(-1, 1)
        y = mid_prices[-lookback:]

        hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]

        estimated_mid_prices *= 1.0007

        bid_spread = pd.Series(self.BIDS[PRODUCT]).values - estimated_mid_prices
        ask_spread = pd.Series(self.ASKS[PRODUCT]).values - estimated_mid_prices

        bid_z_score = (bid_spread - bid_spread.mean()) / bid_spread.std()
        ask_z_score = (ask_spread - ask_spread.mean()) / ask_spread.std()

        z_score_reversal_threshold = 1.75

        long_reversal_entry = ask_z_score[-1] < -z_score_reversal_threshold and ask_z_score[-2] > -z_score_reversal_threshold
        short_reversal_entry = ask_z_score[-1] > z_score_reversal_threshold and ask_z_score[-2] < z_score_reversal_threshold

        exit = current_pos * ask_z_score[-1] > 0 or current_pos * bid_z_score[-1] > 0

        logger.print(bid_z_score[-1], ask_z_score[-1])
        logger.print(long_reversal_entry, short_reversal_entry)
        logger.print(exit, current_pos) 

        for ask, vol in list(ordered_sell_dict.items()):
            if (long_reversal_entry and current_pos < self.LIMITS[PRODUCT]) or (exit and current_pos < 0):
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, ask, order_vol))
                current_pos += order_vol

        for bid, vol in list(ordered_buy_dict.items()):
            if (short_reversal_entry and current_pos > -self.LIMITS[PRODUCT]) or (exit and current_pos > 0):
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, bid, order_vol))
                current_pos += order_vol

        return orders
    
    def compute_orders_jams(self, PRODUCT, order_depth):
        orders: list[Order] = []

        COMPONENTS = ["CROISSANTS", "JAMS"]

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        lookback = 100

        mid_prices = (pd.Series(self.ASKS[PRODUCT]) + pd.Series(self.BIDS[PRODUCT])) / 2
        if len(mid_prices) < lookback:
            return orders

        component_mid_prices = np.array([(pd.Series(self.ASKS[COMPONENT]) + pd.Series(self.BIDS[COMPONENT])) / 2 for COMPONENT in COMPONENTS])
        estimated_mid_prices = component_mid_prices.T @ np.array([4, 2])
        
        X = estimated_mid_prices[-lookback:].reshape(-1, 1)
        y = mid_prices[-lookback:]

        hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]

        estimated_mid_prices *= hedge_ratio

        bid_spread = pd.Series(self.BIDS[PRODUCT]).values - estimated_mid_prices
        ask_spread = pd.Series(self.ASKS[PRODUCT]).values - estimated_mid_prices

        bid_z_score = (bid_spread - bid_spread.mean()) / bid_spread.std()
        ask_z_score = (ask_spread - ask_spread.mean()) / ask_spread.std()

        z_score_reversal_threshold = 1.75

        long_reversal_entry = ask_z_score[-1] > ask_z_score[-2] and ask_z_score[-2] < -z_score_reversal_threshold
        short_reversal_entry = bid_z_score[-1] < bid_z_score[-2] and bid_z_score[-2] > z_score_reversal_threshold

        exit = current_pos * ask_z_score[-1] > 0 or current_pos * bid_z_score[-1] > 0

        logger.print(bid_z_score[-1], ask_z_score[-1])
        logger.print(long_reversal_entry, short_reversal_entry)
        logger.print(exit, current_pos) 

        for ask, vol in list(ordered_sell_dict.items()):
            if (long_reversal_entry and current_pos < self.LIMITS[PRODUCT]) or (exit and current_pos < 0):
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, ask, order_vol))
                current_pos += order_vol

        for bid, vol in list(ordered_buy_dict.items()):
            if (short_reversal_entry and current_pos > -self.LIMITS[PRODUCT]) or (exit and current_pos > 0):
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, bid, order_vol))
                current_pos += order_vol

        return orders
    
    def compute_orders_croissants(self, PRODUCT, order_depth):
        orders: list[Order] = []

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        best_ask = [ask for ask, _ in list(ordered_sell_dict.items())][0]
        best_bid = [bid for bid, _ in list(ordered_buy_dict.items())][0]

        orders.append(Order(PRODUCT, best_ask - 1, int((self.LIMITS[PRODUCT] - current_pos) * .7)))
        orders.append(Order(PRODUCT, best_bid + 1, int((-self.LIMITS[PRODUCT] - current_pos) * .7)))

        return orders
    
    
    def compute_orders_options(self, PRODUCT, order_depth, volatility, strike_price):
        orders: list[Order] = []
        
        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)
        position_limit = self.LIMITS.get(PRODUCT, 200)

        # Get the underlying asset price (VOLCANIC_ROCK)
        if "VOLCANIC_ROCK" in self.state.order_depths:
            underlying_order_depth = self.state.order_depths["VOLCANIC_ROCK"]
            underlying_sells = collections.OrderedDict(sorted(underlying_order_depth.sell_orders.items()))
            underlying_buys = collections.OrderedDict(sorted(underlying_order_depth.buy_orders.items(), reverse=True))
            
            # Use bid/ask instead of mid price for underlying
            if underlying_buys:
                underlying_bid = max(underlying_buys.keys())
            else:
                underlying_bid = None
                
            if underlying_sells:
                underlying_ask = min(underlying_sells.keys())
            else:
                underlying_ask = None
                
            # Use bid for pricing when we're considering selling options (short entry)
            # Use ask for pricing when we're considering buying options (long entry)
            if underlying_bid and underlying_ask:
                underlying_price = underlying_bid  # Default to bid for conservative valuation
            elif underlying_bid:
                underlying_price = underlying_bid
            elif underlying_ask:
                underlying_price = underlying_ask
            else:
                underlying_price = strike_price  # Fallback

        # Calculate option's bid/ask values
        if ordered_buy_dict:
            option_bid = max(ordered_buy_dict.keys())
        else:
            option_bid = None
            
        if ordered_sell_dict:
            option_ask = min(ordered_sell_dict.keys())
        else:
            option_ask = None
        
        # Calculate mid price for reference
        if option_bid and option_ask:
            mid_price = (option_bid + option_ask) / 2
        elif option_bid:
            mid_price = option_bid
        elif option_ask:
            mid_price = option_ask
        else:
            mid_price = None  # Will be filled in later

        expiration_time = 5 / 252  # Approx. time to expiry
        
        # Calculate GARCH volatility if we have enough underlying price history
        if "VOLCANIC_ROCK" in self.BIDS and len(self.BIDS["VOLCANIC_ROCK"]) > 30:
            # Calculate log returns from price series
            prices = pd.Series(self.BIDS["VOLCANIC_ROCK"])
            returns = np.log(prices / prices.shift(1)).dropna().values
            
            # Apply GARCH model to forecast volatility
            garch_volatility = self.fit_garch_and_forecast(returns)
            
            # Use GARCH volatility instead of fixed volatility
            fair_price = self.black_scholes(underlying_price, strike_price, expiration_time, 0, garch_volatility)
        else:
            # Fall back to fixed volatility when insufficient data
            fair_price = self.black_scholes(underlying_price, strike_price, expiration_time, 0, volatility)
        
        # If we couldn't calculate mid_price earlier, use our Black-Scholes estimate
        if mid_price is None:
            mid_price = fair_price
        
        # Determine trading signals based on misprice rate per strike price
        misprice_rate = 1        
        if strike_price == 10500:
            misprice_rate = 0.6  # For lower-delta options
        elif strike_price == 10250:
            misprice_rate = 0.85  # For medium-delta options
        elif strike_price == 10000:
            misprice_rate = 1.25  # For ATM options
        elif strike_price == 9750:
            misprice_rate = 2.0  # For medium-delta options
        elif strike_price == 9500:
            misprice_rate = 2.5  # For higher-delta options
        
        # Use bid/ask for more accurate entry points
        short_condition = fair_price + misprice_rate
        long_condition = fair_price - misprice_rate
        
        # Entry conditions using bid/ask instead of mid price
        short_entry = option_bid and option_bid > short_condition
        long_entry = option_ask and option_ask < long_condition
        
        # Exit conditions using bid/ask for current position
        if current_pos > 0 and option_bid:
            price_diff_pct = (option_bid - fair_price) / fair_price
            if price_diff_pct > 0.0001:  # Small positive threshold
                self.exit = True
                logger.print(f"Exit LONG position for {PRODUCT}: Bid {option_bid:.2f} above theoretical {fair_price:.2f}")
        elif current_pos < 0 and option_ask:
            price_diff_pct = (option_ask - fair_price) / fair_price
            if price_diff_pct < -0.0001:  # Small negative threshold
                self.exit = True
                logger.print(f"Exit SHORT position for {PRODUCT}: Ask {option_ask:.2f} below theoretical {fair_price:.2f}")
        
        # Also exit if we're heavily in a position and the price has moved toward fair value
        if current_pos > 50 and option_bid and option_bid > fair_price:
            self.exit = True
            logger.print(f"Exit large LONG position for {PRODUCT}: Bid {option_bid:.2f} above theoretical {fair_price:.2f}")
        elif current_pos < -50 and option_ask and option_ask < fair_price:
            self.exit = True
            logger.print(f"Exit large SHORT position for {PRODUCT}: Ask {option_ask:.2f} below theoretical {fair_price:.2f}")
        
        # Debug logging
        logger.print(f"{PRODUCT} - Fair: {fair_price:.2f}, Position: {current_pos}, Short Entry: {short_entry}, Long Entry: {long_entry}")
        
        # Execute orders based on signals
        for ask, vol in list(ordered_sell_dict.items()):
            if (long_entry and current_pos < position_limit) or (self.exit and current_pos < 0):
                logger.print(f"Long Entry: {long_condition:.2f}, Fair Price: {fair_price:.2f}, Ask: {ask}")
                order_vol = min(-vol, position_limit - current_pos)
                if self.exit: 
                    order_vol = min(-vol, int(-current_pos * (1/3)))
                orders.append(Order(PRODUCT, ask, order_vol))
                current_pos += order_vol

        for bid, vol in list(ordered_buy_dict.items()):
            if (short_entry and current_pos > -position_limit) or (self.exit and current_pos > 0):
                logger.print(f"Short Entry: {short_condition:.2f}, Fair Price: {fair_price:.2f}, Bid: {bid}")
                order_vol = max(-vol, -position_limit - current_pos)
                if self.exit: 
                    order_vol = max(-vol, int(-current_pos * (1/3)))  # Using max for buy orders
                orders.append(Order(PRODUCT, bid, order_vol))
                current_pos += order_vol

        return orders

    def fit_garch_and_forecast(self, returns, forecast_horizon=1):
        """Fit GARCH(1,1) model and forecast volatility"""
        # Initial parameters [omega, alpha, beta]
        omega, alpha, beta = 0.01, 0.1, 0.8
        
        # Simple implementation - in practice you'd use optimization
        T = len(returns)
        sigma2 = np.zeros(T)
        
        # Initialize with sample variance
        sigma2[0] = np.var(returns)
        
        # Compute the GARCH variances
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        # Forecast next period volatility
        forecast_var = omega + alpha * returns[-1]**2 + beta * sigma2[-1]
        
        # Convert to annualized volatility (assuming returns are daily)
        annualized_vol = np.sqrt(forecast_var * 252)
        
        return annualized_vol
        
    
    def black_scholes(
            self,
            asset_price: float,
            strike_price: float,
            expiration_time: float,
            risk_free_rate: float,
            volatility: float,
        ) -> float:
            d1 = (math.log(asset_price / strike_price) + (risk_free_rate + volatility ** 2 / 2) * expiration_time) / (volatility * math.sqrt(expiration_time))
            d2 = d1 - volatility * math.sqrt(expiration_time)
            
            # Use the correct format for NormalDist.cdf
            from statistics import NormalDist
            normal = NormalDist(mu=0, sigma=1)
            
            return asset_price * normal.cdf(d1) - strike_price * math.exp(-risk_free_rate * expiration_time) * normal.cdf(d2)