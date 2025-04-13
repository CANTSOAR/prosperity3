from typing import List
import string
import collections
import json
from typing import Any
import numpy as np
import pandas as pd

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
        "PICNIC_BASKET2": 100
    }

    BIDS = {
        "RAINFOREST_RESIN": [],
        "KELP": [],
        "SQUID_INK": [],
        "CROISSANTS": [],
        "JAMS": [],
        "DJEMBES": [],
        "PICNIC_BASKET1": [],
        "PICNIC_BASKET2": []
    }

    ASKS = {
        "RAINFOREST_RESIN": [],
        "KELP": [],
        "SQUID_INK": [],
        "CROISSANTS": [],
        "JAMS": [],
        "DJEMBES": [],
        "PICNIC_BASKET1": [],
        "PICNIC_BASKET2": []
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
        }

        self.populate_prices()

        for product in [
            "PICNIC_BASKET1",
            "PICNIC_BASKET2"
        ]:
            order_depth: OrderDepth = state.order_depths[product]
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

        COMPONENTS = ["CROISSANTS", "JAMS", "DJEMBES"]

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        wait = 100

        if len(self.ASKS[PRODUCT]) < wait:
            return orders

        component_mid_prices = np.array([(pd.Series(self.ASKS[COMPONENT]) + pd.Series(self.BIDS[COMPONENT])) / 2 for COMPONENT in COMPONENTS])
        estimated_mid_prices = component_mid_prices.T @ np.array([6, 1, 3])

        estimated_mid_prices *= 1.0008

        bid_spread = pd.Series(self.BIDS[PRODUCT]).values - estimated_mid_prices
        ask_spread = pd.Series(self.ASKS[PRODUCT]).values - estimated_mid_prices

        bid_z_score = (bid_spread[-2:] - bid_spread.mean()) / bid_spread.std()
        ask_z_score = (ask_spread[-2:] - ask_spread.mean()) / ask_spread.std()

        z_score_reversal_threshold = 1.75

        long_reversal_entry = ask_z_score[-1] < -z_score_reversal_threshold and ask_z_score[-2] > -z_score_reversal_threshold
        short_reversal_entry = ask_z_score[-1] > z_score_reversal_threshold and ask_z_score[-2] < z_score_reversal_threshold

        exit = current_pos * ask_z_score[-1] > 0 or current_pos * bid_z_score[-1] > 0

        logger.print(bid_z_score[-1], ask_z_score[-1])
        logger.print(long_reversal_entry, short_reversal_entry)
        logger.print(exit, current_pos) 
        ratio = .1
        
        for ask, vol in list(ordered_sell_dict.items()):
            if (long_reversal_entry and current_pos < self.LIMITS[PRODUCT]) or (exit and current_pos < 0):
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, ask, order_vol))
                current_pos += order_vol

        for COMPONENT, factor in zip(COMPONENTS, [6, 3, 1]):
            if (long_reversal_entry and current_pos < self.LIMITS[PRODUCT]) or (exit and current_pos < 0):
                component_current_pos = self.POSITIONS.get(COMPONENT, 0)
                component_order_depth = self.state.order_depths[COMPONENT]
                component_ordered_buy_dict = collections.OrderedDict(sorted(component_order_depth.buy_orders.items(), reverse=True))

                for component_bid, component_vol in list(component_ordered_buy_dict.items()):
                    component_order_vol = int(max(-component_vol, -self.LIMITS[COMPONENT] - component_current_pos) * ratio * factor)
                    if exit: component_order_vol = int(max(-component_vol, -component_current_pos) * ratio * factor)
                    logger.print(COMPONENT, component_bid, component_order_vol, component_ordered_buy_dict, component_order_depth)
                    orders.append(Order(COMPONENT, component_bid, component_order_vol))
                    component_current_pos += component_order_vol

        for bid, vol in list(ordered_buy_dict.items()):
            if (short_reversal_entry and current_pos > -self.LIMITS[PRODUCT]) or (exit and current_pos > 0):
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = max(-vol, -current_pos)
                orders.append(Order(PRODUCT, bid, order_vol))
                current_pos += order_vol
                
        for COMPONENT, factor in zip(COMPONENTS, [6, 3, 1]):
            if (short_reversal_entry and current_pos > -self.LIMITS[PRODUCT]) or (exit and current_pos > 0):
                component_current_pos = self.POSITIONS.get(COMPONENT, 0)
                component_order_depth = self.state.order_depths[COMPONENT]
                component_ordered_sell_dict = collections.OrderedDict(sorted(component_order_depth.sell_orders.items()))

                for component_ask, component_vol in list(component_ordered_sell_dict.items()):
                    component_order_vol = int(min(-component_vol, self.LIMITS[COMPONENT] - component_current_pos) * ratio * factor)
                    if exit: component_order_vol = int(min(-component_vol, -component_current_pos) * ratio * factor)
                    logger.print(COMPONENT, component_ask, component_order_vol)
                    orders.append(Order(COMPONENT, component_ask, component_order_vol))
                    component_current_pos += component_order_vol

        return orders
   
    
    def compute_orders_basket_2(self, PRODUCT, order_depth):
        orders: list[Order] = []

        COMPONENTS = ["CROISSANTS", "JAMS"]

        ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_pos = self.POSITIONS.get(PRODUCT, 0)

        wait = 100

        if len(self.ASKS[PRODUCT]) < wait:
            return orders

        component_mid_prices = np.array([(pd.Series(self.ASKS[COMPONENT]) + pd.Series(self.BIDS[COMPONENT])) / 2 for COMPONENT in COMPONENTS])
        estimated_mid_prices = component_mid_prices.T @ np.array([4, 2])

        estimated_mid_prices *= 1.0007

        bid_spread = pd.Series(self.BIDS[PRODUCT]).values - estimated_mid_prices
        ask_spread = pd.Series(self.ASKS[PRODUCT]).values - estimated_mid_prices

        bid_z_score = (bid_spread[-2:] - bid_spread.mean()) / bid_spread.std()
        ask_z_score = (ask_spread[-2:] - ask_spread.mean()) / ask_spread.std()

        z_score_reversal_threshold = 1.75

        long_reversal_entry = ask_z_score[-1] < -z_score_reversal_threshold and ask_z_score[-2] > -z_score_reversal_threshold
        short_reversal_entry = ask_z_score[-1] > z_score_reversal_threshold and ask_z_score[-2] < z_score_reversal_threshold

        exit = current_pos * ask_z_score[-1] > 0 or current_pos * bid_z_score[-1] > 0

        logger.print(bid_z_score[-1], ask_z_score[-1])
        logger.print(long_reversal_entry, short_reversal_entry)
        logger.print(exit, current_pos) 
        ratio = .1

        for ask, vol in list(ordered_sell_dict.items()):
            if (long_reversal_entry and current_pos < self.LIMITS[PRODUCT]) or (exit and current_pos < 0):
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = min(-vol, -current_pos)
                orders.append(Order(PRODUCT, ask, order_vol))
                current_pos += order_vol

        for COMPONENT, factor in zip(COMPONENTS, [4, 2]):
            if (long_reversal_entry and current_pos < self.LIMITS[PRODUCT]) or (exit and current_pos < 0):
                component_current_pos = self.POSITIONS.get(COMPONENT, 0)
                component_order_depth = self.state.order_depths[COMPONENT]
                component_ordered_buy_dict = collections.OrderedDict(sorted(component_order_depth.buy_orders.items(), reverse=True))

                for component_bid, component_vol in list(component_ordered_buy_dict.items()):
                    component_order_vol = int(max(-component_vol, -self.LIMITS[COMPONENT] - component_current_pos) * ratio * factor)
                    if exit: component_order_vol = int(max(-component_vol, -component_current_pos) * ratio * factor)
                    logger.print(COMPONENT, component_bid, component_order_vol, component_ordered_buy_dict, component_order_depth)
                    orders.append(Order(COMPONENT, component_bid, component_order_vol))
                    component_current_pos += component_order_vol

        for bid, vol in list(ordered_buy_dict.items()):
            if (short_reversal_entry and current_pos > -self.LIMITS[PRODUCT]) or (exit and current_pos > 0):
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos)
                if exit: order_vol = max(-vol, -current_pos)
                orders.append(Order(PRODUCT, bid, order_vol))
                current_pos += order_vol

        for COMPONENT, factor in zip(COMPONENTS, [4, 2]):
            if (short_reversal_entry and current_pos > -self.LIMITS[PRODUCT]) or (exit and current_pos > 0):
                component_current_pos = self.POSITIONS.get(COMPONENT, 0)
                component_order_depth = self.state.order_depths[COMPONENT]
                component_ordered_sell_dict = collections.OrderedDict(sorted(component_order_depth.sell_orders.items()))

                for component_ask, component_vol in list(component_ordered_sell_dict.items()):
                    component_order_vol = int(min(-component_vol, self.LIMITS[COMPONENT] - component_current_pos) * ratio * factor)
                    if exit: component_order_vol = int(min(-component_vol, -component_current_pos) * ratio * factor)
                    logger.print(COMPONENT, component_ask, component_order_vol)
                    orders.append(Order(COMPONENT, component_ask, component_order_vol))
                    component_current_pos += component_order_vol

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