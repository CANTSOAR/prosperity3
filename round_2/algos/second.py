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


	
	exit = 0
	marketmaking = False 
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
	
	def run(self, state: TradingState):
		# Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
		result = {}
		self.state = state
		self.POSITIONS = state.position

		COMPUTE_ORDERS = {
			"RAINFOREST_RESIN": self.compute_orders_resin,
			"SQUID_INK": self.compute_orders_ink,
			"KELP": self.compute_orders_kelp,
			"PICNIC_BASKET1": self.compute_orders_basket_1
		}

		for product in [
			"PICNIC_BASKET1"
		]:
			order_depth: OrderDepth = state.order_depths[product]
			orders = self.compute_orders_basket_1(product, order_depth)
			
			#result[product] = COMPUTE_ORDERS[product](product,order_depth) + orders
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
		for COMPONENT in COMPONENTS:
			component_order_depth = self.state.order_depths[COMPONENT]

			ordered_sell_dict = collections.OrderedDict(sorted(component_order_depth.sell_orders.items()))
			ordered_buy_dict = collections.OrderedDict(sorted(component_order_depth.buy_orders.items(), reverse=True))

			current_pos = self.POSITIONS.get(COMPONENT, 0)

			best_ask = [ask for ask, _ in ordered_sell_dict.items()][0]
			best_bid = [bid for bid, _ in ordered_buy_dict.items()][0]

			self.ASKS[COMPONENT].append(best_ask)
			self.BIDS[COMPONENT].append(best_bid)

		component_order_depth = self.state.order_depths[PRODUCT]

		ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
		ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

		current_pos = self.POSITIONS.get(PRODUCT, 0)

		best_ask = [ask for ask, _ in ordered_sell_dict.items()][0]
		best_bid = [bid for bid, _ in ordered_buy_dict.items()][0]

		self.ASKS[PRODUCT].append(best_ask)
		self.BIDS[PRODUCT].append(best_bid)

		mid_prices = (pd.Series(self.ASKS[PRODUCT]) + pd.Series(self.BIDS[PRODUCT])) / 2

		
		hedge_ratio = 1.00082419531
		mean_spread = .0547467078
		std_spread = 83.5605701

		component_mid_prices = np.array([(pd.Series(self.ASKS[COMPONENT]) + pd.Series(self.BIDS[COMPONENT])) / 2 for COMPONENT in COMPONENTS])
		estimated_mid_prices = component_mid_prices.T @ np.array([6, 3, 1]) * hedge_ratio

		spread = mid_prices.values - estimated_mid_prices
		z_score = (spread - mean_spread) / std_spread

		z_score_threshold = 1.3

		long_entry = mid_prices[-1] < .9995 * estimated_mid_prices[-1]
		short_entry = mid_prices[-1] > 1.0005 * estimated_mid_prices[-1]
		
		logger.print(z_score[-1])
		
		lastexit = self.exit
		
		if long_entry or short_entry:
			self.marketmaking = False
		elif current_pos == 0:
			self.marketmaking = True
		
		if ((current_pos > 0 and z_score[-1] >= 0) or (current_pos < 0 and z_score[-1] <= 0)) and self.marketmaking == False:               
			self.exit = True
		else:
			self.exit = False

		for ask, vol in ordered_sell_dict.items():
			
			# This is the exit for if we are short, if yesterday we couldn't fully liquidate, we try and liquidate today
			if (self.exit and current_pos < 0):
				logger.print("EXIT AND GO LONG")
				order_vol = min(vol,-current_pos)
				orders.append(Order(PRODUCT, ask, -order_vol))
				current_pos += order_vol
			elif (lastexit and current_pos < 0):
				logger.print("EXIT AND GO LONG LIQUIDATE")
				order_vol = -current_pos
				orders.append(Order(PRODUCT, ask, order_vol))
				current_pos += order_vol
				
				
			if long_entry and current_pos < self.LIMITS[PRODUCT]:
				logger.print("LONG")
				max_allowed_volume = self.LIMITS[PRODUCT] - current_pos
				logger.print(current_pos)
				order_vol = min(-vol, max_allowed_volume)  # Take only available volume
				logger.print(order_vol)
				if order_vol > 0:
					# Accepting existing ask instead of making a new order
					orders.append(Order(PRODUCT, ask, order_vol))
				current_pos += order_vol
		
		
		for bid, vol in ordered_buy_dict.items():
			
			
			# This is the exit for if we are long, if yesterday we couldn't fully liquidate, we try and liquidate today
			if (self.exit and current_pos > 0):
				logger.print("EXIT AND GO SHORT")
				order_vol = min(vol,-current_pos)
				orders.append(Order(PRODUCT, bid, -order_vol))
				current_pos += order_vol
			elif (lastexit and current_pos > 0):
				logger.print("EXIT AND GO SHORT LIQUIDATE")
				order_vol = -current_pos
				orders.append(Order(PRODUCT, bid, order_vol))
				current_pos += order_vol
				
				
			if short_entry and current_pos > -self.LIMITS[PRODUCT]:
				logger.print("SHORT")                   
				max_allowed_volume = -self.LIMITS[PRODUCT] - current_pos
				logger.print(current_pos)
				order_vol = max(-vol, max_allowed_volume)  # Take only available volume
				logger.print(order_vol)
				if order_vol < 0:
					# Accepting existing bid instead of making a new order
					orders.append(Order(PRODUCT, bid, order_vol))
				current_pos += order_vol
		return orders		
		
  
  
	def market_making(self, PRODUCT, order_depth):
		orders: list[Order] = []
		# Fixed: Using order_depth directly instead of component_order_depth
		ordered_sell_dict = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
		ordered_buy_dict = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

		current_pos = self.POSITIONS.get(PRODUCT, 0)

		# Safety check for empty order books
		if not ordered_sell_dict or not ordered_buy_dict:
			logger.print(f"Warning: Empty order book for {PRODUCT}, skipping market making")
			return orders
			
		best_ask = list(ordered_sell_dict.keys())[0]
		best_bid = list(ordered_buy_dict.keys())[0]

		# Market making logic
		logger.print("MarketMaking")

		best_remaining_ask = best_ask
		best_remaining_bid = best_bid
		current_spread = best_remaining_ask - best_remaining_bid
			
		logger.print(f"Best ask: {best_remaining_ask}, Best bid: {best_remaining_bid}, Spread: {current_spread}")
		logger.print(f"Current position: {current_pos}")

		# Market making parameters
		undercut_amount = 1
		base_order_size = 10
		min_profitable_spread = undercut_amount * 2 + 1

		# Position management settings
		target_position = 0  # We prefer to be neutral
		position_threshold = 3  # When to start taking more aggressive action
		critical_threshold = 8  # When to take extreme measures

		# Check if we have a significant position that needs to be reduced
		has_significant_position = abs(current_pos) > position_threshold
		has_critical_position = abs(current_pos) > critical_threshold

		# If position is critical, prioritize liquidation over normal market making
		if has_critical_position:
			logger.print(f"CRITICAL POSITION DETECTED: {current_pos}. Prioritizing liquidation.")
			
			if current_pos > 0:  # Long position - aggressive selling
				# Place multiple sell orders at and below the bid to ensure they get filled
				sell_price = best_remaining_bid
				# Use larger size for critical liquidation
				sell_size = min(current_pos, int(base_order_size * 3.0))
				orders.append(Order(PRODUCT, sell_price, -sell_size))
				logger.print(f"CRITICAL position reduction: Selling {sell_size} @ {sell_price}")
				
				# If position is very large, place another order at an even more aggressive price
				remaining_position = current_pos - sell_size
				if remaining_position > 0:
					sell_price_aggressive = best_remaining_bid - 2  # Even more aggressive price
					sell_size_aggressive = min(remaining_position, int(base_order_size * 2.0))
					orders.append(Order(PRODUCT, sell_price_aggressive, -sell_size_aggressive))
					logger.print(f"CRITICAL position reduction (second level): Selling {sell_size_aggressive} @ {sell_price_aggressive}")
			
			elif current_pos < 0:  # Short position - aggressive buying
				# Place multiple buy orders at and above the ask to ensure they get filled
				buy_price = best_remaining_ask
				# Use larger size for critical liquidation
				buy_size = min(abs(current_pos), int(base_order_size * 3.0))
				orders.append(Order(PRODUCT, buy_price, buy_size))
				logger.print(f"CRITICAL position reduction: Buying {buy_size} @ {buy_price}")
				
				# If position is very large, place another order at an even more aggressive price
				remaining_position = abs(current_pos) - buy_size
				if remaining_position > 0:
					buy_price_aggressive = best_remaining_ask + 2  # Even more aggressive price
					buy_size_aggressive = min(remaining_position, int(base_order_size * 2.0))
					orders.append(Order(PRODUCT, buy_price_aggressive, buy_size_aggressive))
					logger.print(f"CRITICAL position reduction (second level): Buying {buy_size_aggressive} @ {buy_price_aggressive}")
			
			# Return early to skip normal market making logic
			return orders

		# Normal market making logic continues below for non-critical positions
		# Skew prices to favor position reduction
		if has_significant_position:
			if current_pos > 0:  # Long position - skew prices to favor selling
				sell_price_skew = -1  # More aggressive sell price (lower)
				buy_price_skew = -2   # Less aggressive buy price (lower)
			else:  # Short position - skew prices to favor buying
				sell_price_skew = 2   # Less aggressive sell price (higher)
				buy_price_skew = 1    # More aggressive buy price (higher)
		else:
			sell_price_skew = 0
			buy_price_skew = 0

		# Only market make if spread is sufficiently wide
		if current_spread > min_profitable_spread:
			# SELL SIDE - prioritize if we're long
			sell_price = best_remaining_ask - undercut_amount + sell_price_skew
			
			# Adjust size based on current position
			if current_pos > position_threshold:  # We're long, need to reduce
				# Larger size to reduce position
				sell_size_factor = 2.0
				logger.print("Position reduction mode: SELLING")
			elif current_pos < -position_threshold:  # We're short, be cautious with selling
				# Smaller size to avoid increasing short position
				sell_size_factor = 0.5
			else:  # Normal market making
				sell_size_factor = 1.0
			
			# Calculate sell order size with minimum check
			sell_size = min(
				max(1, int(base_order_size * sell_size_factor)),
				self.LIMITS[PRODUCT] + current_pos  # Don't exceed short limit
			)
			
			# Only place sell order if we're not at position limit
			if current_pos > -self.LIMITS[PRODUCT] and sell_size > 0:
				# Increase size if we're trying to reduce a long position
				if current_pos > position_threshold:
					sell_size = min(sell_size * 2, current_pos)  # Sell up to current position
				
				orders.append(Order(PRODUCT, sell_price, -sell_size))
				logger.print(f"Placing sell order: {sell_size} @ {sell_price}")
			
			# BUY SIDE - prioritize if we're short
			buy_price = best_remaining_bid + undercut_amount + buy_price_skew
			
			# Adjust size based on current position
			if current_pos < -position_threshold:  # We're short, need to reduce
				# Larger size to reduce position
				buy_size_factor = 2.0
				logger.print("Position reduction mode: BUYING")
			elif current_pos > position_threshold:  # We're long, be cautious with buying
				# Smaller size to avoid increasing long position
				buy_size_factor = 0.5
			else:  # Normal market making
				buy_size_factor = 1.0
			
			# Calculate buy order size with minimum check
			buy_size = min(
				max(1, int(base_order_size * buy_size_factor)),
				self.LIMITS[PRODUCT] - current_pos  # Don't exceed long limit
			)
			
			# Only place buy order if we're not at position limit
			if current_pos < self.LIMITS[PRODUCT] and buy_size > 0:
				# Increase size if we're trying to reduce a short position
				if current_pos < -position_threshold:
					buy_size = min(buy_size * 2, abs(current_pos))  # Buy up to current position
				
				orders.append(Order(PRODUCT, buy_price, buy_size))
				logger.print(f"Placing buy order: {buy_size} @ {buy_price}")
		else:
			logger.print(f"Spread too narrow ({current_spread}) for profitable market making")

		# If we have a significant position, place an aggressive order to reduce it
		if has_significant_position and len(orders) == 0:
			if current_pos > 0:  # Long position - try to sell
				# Place sell order at the bid to ensure it gets filled
				sell_price = best_remaining_bid
				sell_size = min(current_pos, int(base_order_size * 1.5))
				orders.append(Order(PRODUCT, sell_price, -sell_size))
				logger.print(f"URGENT position reduction: Selling {sell_size} @ {sell_price}")
			elif current_pos < 0:  # Short position - try to buy
				# Place buy order at the ask to ensure it gets filled
				buy_price = best_remaining_ask
				buy_size = min(abs(current_pos), int(base_order_size * 1.5))
				orders.append(Order(PRODUCT, buy_price, buy_size))
				logger.print(f"URGENT position reduction: Buying {buy_size} @ {buy_price}")

		return orders