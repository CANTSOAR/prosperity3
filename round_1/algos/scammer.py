from typing import List
import string
import collections
import json
from typing import Any
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

    POSITIONS = {
        "RAINFOREST_RESIN": 0,
        "KELP": 0,
        "SQUID_INK": 0
    }

    LIMITS = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50
    }

    resin_short = 0
    resin_long = 0

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
        for product in ["RAINFOREST_RESIN"]:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acceptable_bid = 10000, acceptable_ask = 10000, undercut_amount = 0)
            
            result[product] = orders
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    def compute_orders(self, PRODUCT, order_depth, acceptable_bid, acceptable_ask, undercut_amount):
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

        current_pos = self.POSITIONS[PRODUCT]

        orders.append(Order(PRODUCT, acceptable_bid - 4, 10)) # BUY order to recuperate at lower price
        orders.append(Order(PRODUCT, acceptable_ask + 4, -10)) # SELL order to drop at higher price

        for ask, vol in ordered_sell_dict.items():
            if ask <= acceptable_ask and current_pos < self.LIMITS[PRODUCT]:
                order_vol = min(-vol, self.LIMITS[PRODUCT] - current_pos) # take the minimum of available volume and the volume we are allowed to take
                orders.append(Order(PRODUCT, ask + undercut_amount, order_vol)) # this is a BUY order, we undercut by paying a little MORE
                current_pos += order_vol\

        for bid, vol in ordered_buy_dict.items():
            if bid >= acceptable_bid and current_pos > -self.LIMITS[PRODUCT]:
                order_vol = max(-vol, -self.LIMITS[PRODUCT] - current_pos) # take the minimum of available volume and the volume we are allowed to take
                orders.append(Order(PRODUCT, bid - undercut_amount, order_vol)) # this is a SELL order, we undercut by selling a bit CHEAPER
                current_pos += order_vol
                
        return orders