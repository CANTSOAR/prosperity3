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
        "SQUID_INK": 50
    }

    BIDS = {
        "RAINFOREST_RESIN": [],
        "KELP": [2028],
        "SQUID_INK": []
    }

    ASKS = {
        "RAINFOREST_RESIN": [],
        "KELP": [2032],
        "SQUID_INK": []
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
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        result = {}
        self.POSITIONS = state.position

        COMPUTE_ORDERS = {
            "RAINFOREST_RESIN": self.compute_orders_resin,
            "SQUID_INK": self.compute_orders_ink,
            "KELP": self.compute_orders_kelp
        }

        for product in [
            "RAINFOREST_RESIN",
            "SQUID_INK",
            "KELP"
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
            self.ASKS[PRODUCT].append(self.ASKS[PRODUCT][-1])
            self.kelp_last_ask = best_ask

        if best_bid == self.kelp_last_bid:
            self.BIDS[PRODUCT].append(best_bid)
        else:
            self.BIDS[PRODUCT].append(self.BIDS[PRODUCT][-1])
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

            #moving_average, standard_dev = self.kalman_mean_std(mid_prices, q = .01, r = .1, alpha = .15)
            #mid_prices = mid_prices.values

            #band_z_score = 1.5
            #upper_band = moving_average + band_z_score * standard_dev
            #lower_band = moving_average - band_z_score * standard_dev

            if standard_dev.values[-1] < 1:
                return orders

            z_score = (mid_prices.values[-1] - moving_average.values[-1]) / standard_dev.values[-1]
            prev_z_score = (mid_prices.values[-2] - moving_average.values[-2]) / standard_dev.values[-2]

            z_score_thresh = 1.7
            z_score_severe_thresh = 2.25

            logger.print(f"z-score: {z_score}")

            for ask, vol in list(ordered_sell_dict.items())[:2]:
                
                z_cross = z_score < -z_score_thresh and prev_z_score < -z_score_thresh
                severe_cross = z_score < z_score_severe_thresh

                if (z_cross or severe_cross) and current_pos < self.LIMITS[PRODUCT]:
                    order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos)
                    orders.append(Order(PRODUCT, ask, order_vol))
                    current_pos += order_vol

                    if severe_cross:
                        self.ink_lock = True
                    else:
                        self.ink_lock = False

                ask_cross_mid = z_score < 0

                if ask_cross_mid and current_pos < 0 and not self.ink_lock:
                    order_vol = min(-vol, -current_pos)
                    orders.append(Order(PRODUCT, ask, order_vol))
                    current_pos += order_vol

            for bid, vol in list(ordered_buy_dict.items())[:2]:

                z_cross = z_score > z_score_thresh and prev_z_score > z_score_thresh
                severe_cross = z_score > z_score_severe_thresh

                if (z_cross or severe_cross) and current_pos > -self.LIMITS[PRODUCT]:
                    order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos)
                    orders.append(Order(PRODUCT, bid, order_vol))
                    current_pos += order_vol

                    if severe_cross:
                        self.ink_lock = True
                    else:
                        self.ink_lock = False

                bid_cross_mid = z_score > 0

                if bid_cross_mid and current_pos > 0 and not self.ink_lock:
                    order_vol = max(-vol, -current_pos)
                    orders.append(Order(PRODUCT, bid, order_vol))
                    current_pos += order_vol

        return orders
    
    def kalman_mean_std(self, prices: pd.Series, q=0.001, r=1.0, alpha=0.05):

        n = len(prices)
        mu = np.zeros(n)
        P = np.zeros(n)
        K = np.zeros(n)

        mu[0] = prices.iloc[0]
        P[0] = 1.0
        smoothed_std = np.zeros(n)
        smoothed_std[0] = 0.0

        for k in range(1, n):
            # Predict
            mu_pred = mu[k-1]
            P_pred = P[k-1] + q

            # Kalman Gain
            K[k] = P_pred / (P_pred + r)

            # Update
            mu[k] = mu_pred + K[k] * (prices.iloc[k] - mu_pred)
            P[k] = (1 - K[k]) * P_pred

            # Update smoothed std dev using exponential moving average of squared error
            residual = prices.iloc[k] - mu[k]
            smoothed_std[k] = np.sqrt((1 - alpha) * smoothed_std[k-1]**2 + alpha * residual**2)

        return mu, smoothed_std