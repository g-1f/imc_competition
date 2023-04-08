from typing import Any, Dict, List, Deque
import json
from datamodel import OrderDepth, TradingState, Order, Trade, ProsperityEncoder
import math
import numpy as np
import pandas as pd


class Trader:
    def __init__(self):
        self.position_limits = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            "BERRIES": 250,
            "DIVING_GEAR": 50,
            "DOLPHIN_SIGHTINGS": 0,
            "BAGUETTE": 150,
            "DIP": 300,
            "UKULELE": 70,
            "PICNIC_BASKET": 70,
        }
        self.min_spread_prl = 4
        self.min_spread_bnn = 1.5
        self.berries_start_price = None
        self.berries_stddev = 10
        self.berries_ma = Deque(maxlen=200)
        self.ds_stddev = 0.5
        self.black_swan_triggered = False
        self.last_obs = None
        self.berries_entry = False
        self.berries_exit = False
        self.bnn_prev_price = None
        self.berries_entry_morning = False
        self.berries_exit_morning = False
        self.pack_sum = Deque(maxlen=200)
        self.pic_wmid = Deque(maxlen=200)
        self.long_spread = False
        self.short_spread = False
        self.rolling_spread = Deque(maxlen=200)
        self.alpha = False
        self.position_filled = False

    def get_weighted_mid_price(self, symbol, state):
        order_depth = state.order_depths[symbol]
        avg_ask = sum(
            [
                order_depth.sell_orders[key] * key
                for key in order_depth.sell_orders.keys()
            ]
        ) / sum(order_depth.sell_orders.values())
        avg_bid = sum(
            [order_depth.buy_orders[key] * key for key in order_depth.buy_orders.keys()]
        ) / sum(order_depth.buy_orders.values())
        wmid = (avg_ask + avg_bid) / 2
        return wmid

    def get_position_limit(self, symbol, state):
        bid_1, ask_1, bid_1_vol, ask_1_vol = self.get_bid_ask_info(symbol, state)
        current_position = state.position.get(symbol, 0)
        max_buy = self.position_limits[symbol] - current_position
        max_sell = -self.position_limits[symbol] - current_position
        return min(bid_1_vol, max_buy), max(max_sell, ask_1_vol)

    def quote_bid_ask(self, symbol: str, state: TradingState, orders):
        order_depth: OrderDepth = state.order_depths[symbol]
        position = state.position.get(symbol, 0)
        position_limit = self.position_limits[symbol]
        buy_total = 0
        sell_total = 0
        max_buy_inventory = position_limit - position
        max_sell_inventory = -position_limit - position

        if symbol == "PEARLS":
            for ask, volume in order_depth.sell_orders.items():
                if ask < 10000:
                    volume = abs(volume)
                    if buy_total + position + volume > position_limit:
                        volume = position_limit - position - buy_total
                    if volume > 0:
                        buy_total += volume
                        orders[symbol].append(Order(symbol, ask, volume))

            for bid, volume in order_depth.buy_orders.items():
                if bid > 10000:
                    volume = -abs(volume)
                    if volume + position + sell_total < -position_limit:
                        volume = -position_limit - position - sell_total
                    if volume < 0:
                        sell_total += volume
                        orders[symbol].append(Order(symbol, bid, volume))

            spread = self.min_spread_prl
            buy_remaining = max_buy_inventory - buy_total
            sell_remaining = -sell_total + max_sell_inventory

            if buy_remaining > 0:
                orders[symbol].append(Order(symbol, 10000 - spread, buy_remaining))
            if sell_remaining < 0:
                orders[symbol].append(Order(symbol, 10000 + spread, sell_remaining))

        elif symbol == "BANANAS":
            wmid = self.get_weighted_mid_price(symbol, state)
            position = state.position.get(symbol, 0)
            ask_price = wmid + self.min_spread_bnn
            bid_price = wmid - self.min_spread_bnn
            bid_order = Order(symbol, bid_price, max_buy_inventory)
            ask_order = Order(symbol, ask_price, max_sell_inventory)
            orders[symbol].append(bid_order)
            orders[symbol].append(ask_order)

    def get_bid_ask_info(self, symbol, state):
        order_depth = state.order_depths[symbol]
        bid_1 = (
            max(state.order_depths[symbol].buy_orders)
            if len(state.order_depths[symbol].buy_orders) > 0
            else 0
        )
        bid_1_vol = order_depth.buy_orders[bid_1]
        ask_1 = (
            min(state.order_depths[symbol].sell_orders)
            if len(state.order_depths[symbol].sell_orders) > 0
            else 0
        )
        ask_1_vol = order_depth.sell_orders[ask_1]
        return bid_1, ask_1, bid_1_vol, ask_1_vol

    def taking_strategy(self, state, orders):
        morning_end = 350000
        noon_end = 500000
        noon = 351500
        night = 503000
        bid_1, ask_1, bid_1_vol, ask_1_vol = self.get_bid_ask_info("BERRIES", state)

        wmid = self.get_weighted_mid_price("BERRIES", state)
        self.berries_ma.append(wmid)
        ma_200 = np.mean(self.berries_ma) if len(self.berries_ma) > 1 else wmid

        max_buy_inventory, max_sell_inventory = self.get_position_limit(
            "BERRIES", state
        )
        if self.berries_start_price is None:
            self.berries_start_price = wmid

        if state.timestamp < morning_end:  # morning
            if ma_200 > wmid:
                self.berries_entry_morning = True
                self.berries_exit_morning = False
            if ma_200 < wmid:
                self.berries_entry_morning = False
                self.berries_exit_morning = True

            if self.berries_entry_morning and not self.berries_exit_morning:
                if max_buy_inventory > 0:
                    order_volume = min(bid_1_vol, max_buy_inventory)
                    orders["BERRIES"].append(Order("BERRIES", wmid, order_volume))
                    max_buy_inventory -= order_volume

            if self.berries_exit_morning and not self.berries_entry_morning:
                if max_sell_inventory < 0:
                    orders["BERRIES"].append(
                        Order("BERRIES", wmid, max(ask_1_vol, max_sell_inventory))
                    )
            if max_buy_inventory > 0:
                orders["BERRIES"].append(
                    Order(
                        "BERRIES",
                        self.berries_start_price - 3 * self.berries_stddev,
                        min(bid_1_vol, max_buy_inventory),
                    )
                )

        if noon_end >= state.timestamp > morning_end:
            if ma_200 > wmid:
                self.berries_entry = True

            if self.berries_entry:
                if max_buy_inventory > 0:
                    orders["BERRIES"].append(Order("BERRIES", wmid, max_buy_inventory))

            if state.timestamp > noon and not self.berries_entry:
                if max_buy_inventory > 0:
                    orders["BERRIES"].append(Order("BERRIES", wmid, max_buy_inventory))

        if state.timestamp > noon_end:
            if ma_200 - wmid > 1:
                self.berries_exit = True

            if self.berries_exit:
                if max_sell_inventory < 0:
                    orders["BERRIES"].append(Order("BERRIES", wmid, max_sell_inventory))

            if state.timestamp > night and ma_200 < wmid:
                if max_sell_inventory < 0:
                    # emergency liquidate all
                    orders["BERRIES"].append(Order("BERRIES", wmid, max_sell_inventory))

    def calculate_fair_price_and_position(self, symbol, state):
        wmid = self.get_weighted_mid_price(symbol, state)

        if symbol == "COCONUTS":
            other_symbol = "PINA_COLADAS"
            conversion_rate = 8 / 15
        else:
            other_symbol = "COCONUTS"
            conversion_rate = 15 / 8

        other_wmid = self.get_weighted_mid_price(other_symbol, state)
        fair_price = (wmid + other_wmid * conversion_rate) / 2
        max_buy, max_sell = self.get_position_limit(symbol, state)
        bid_1, ask_1, bid_1_vol, ask_1_vol = self.get_bid_ask_info(symbol, state)
        return (
            fair_price,
            wmid,
            max_buy,
            max_sell,
            bid_1_vol,
            ask_1_vol,
        )

    def get_position(self, symbol, edge, max_buy, max_sell):
        d = {
            "COCONUTS": 30,
            "PINA_COLADAS": 60,
        }
        d_min = {
            "COCONUTS": 2,
            "PINA_COLADAS": 12,
        }
        max_inventory_pct = 1
        min_inventory_pct = 0
        slope = (max_inventory_pct - min_inventory_pct) / (d[symbol] - d_min[symbol])
        intercept = min_inventory_pct - slope * d_min[symbol]

        inventory_pct = slope * abs(edge) + intercept
        inventory_pct = max(0, min(1, inventory_pct))
        position = 0
        if np.sign(edge) > 0:
            position = inventory_pct * max_buy
        elif np.sign(edge) < 0:
            position = inventory_pct * max_sell
        return position

    def rv(self, state, orders):
        symbols = ["COCONUTS", "PINA_COLADAS"]
        for symbol in symbols:
            (
                fair_price,
                wmid,
                max_buy,
                max_sell,
                bid_1_vol,
                ask_1_vol,
            ) = self.calculate_fair_price_and_position(symbol, state)

            window_size = 10
            max_buy = min(max_buy / window_size, bid_1_vol)
            max_sell = max(max_sell / window_size, ask_1_vol)

            edge = fair_price - wmid
            position = self.get_position(symbol, edge, max_buy, max_sell)
            orders[symbol].append(Order(symbol, fair_price, position))

    def black_swan(self, state, orders):
        ds = state.observations["DOLPHIN_SIGHTINGS"]

        if self.last_obs is not None:
            ds_diff = ds - self.last_obs
        else:
            ds_diff = 0
        bid_1 = self.get_bid_price("DIVING_GEAR", state)
        ask_1 = self.get_ask_price("DIVING_GEAR", state)
        available_buy = self.get_available_position("DIVING_GEAR", state)[0]
        available_sell = self.get_available_position("DIVING_GEAR", state)[1]

        if abs(ds_diff) > 12 * self.ds_stddev:
            # Set the black swan flag to True
            self.black_swan_triggered = True

        if self.black_swan_triggered:
            if np.sign(ds_diff) > 0:
                if available_buy > 0:
                    # aggressive taking
                    orders["DIVING_GEAR"].append(
                        Order("DIVING_GEAR", ask_1, available_buy)
                    )
            elif np.sign(ds_diff) < 0:
                if available_sell < 0:
                    orders["DIVING_GEAR"].append(
                        Order("DIVING_GEAR", bid_1, available_sell)
                    )

        self.last_obs = ds

    def get_bid_vol(self, symbol, state):
        bid_1_vol = self.get_bid_ask_info(symbol, state)[2]
        return bid_1_vol

    def get_ask_vol(self, symbol, state):
        ask_1_vol = self.get_bid_ask_info(symbol, state)[3]
        return ask_1_vol

    def get_ask_price(self, symbol, state):
        return self.get_bid_ask_info(symbol, state)[1]

    def get_bid_price(self, symbol, state):
        return self.get_bid_ask_info(symbol, state)[0]

    def get_available_position(self, symbol, state):
        current_position = state.position.get(symbol, 0)
        max_buy = self.position_limits[symbol] - current_position
        max_sell = -self.position_limits[symbol] - current_position
        return max_buy, max_sell

    def rv_2(self, state, orders):
        symbols = ["BAGUETTE", "DIP", "UKULELE", "PICNIC_BASKET"]
        wmid_prices = {
            "BAGUETTE": self.get_weighted_mid_price("BAGUETTE", state),
            "DIP": self.get_weighted_mid_price("DIP", state),
            "UKULELE": self.get_weighted_mid_price("UKULELE", state),
            "PICNIC_BASKET": self.get_weighted_mid_price("PICNIC_BASKET", state),
        }
        pack = (
            2 * wmid_prices["BAGUETTE"]
            + 4 * wmid_prices["DIP"]
            + wmid_prices["UKULELE"]
        )
        self.pack_sum.append(pack)
        self.pic_wmid.append(wmid_prices["PICNIC_BASKET"])
        available_positions = {
            symbol: self.get_available_position(symbol, state) for symbol in symbols
        }

        def get_spread_rolling_beta(X, Y):
            x = np.array(X)
            y = np.array(Y)
            rolling_beta = np.sum(x * y) / np.sum(x**2)
            spread = X - np.array(rolling_beta) * Y
            return spread[-1]

        if len(self.pic_wmid) >= 30:
            spread = get_spread_rolling_beta(self.pic_wmid, self.pack_sum)
            self.rolling_spread.append(spread)
        if len(self.rolling_spread) >= 100:
            diff = spread - sum(self.rolling_spread) / len(self.rolling_spread)
        else:
            diff = 0

        def get_position(symbol, diff):
            d_max_edge = {
                "BAGUETTE": 550,
                "DIP": 550,
                "UKULELE": 550,
                "PICNIC_BASKET": 1000,
            }
            d_min_edge = {"BAGUETTE": 10, "DIP": 20, "UKULELE": 50, "PICNIC_BASKET": 50}
            max_inventory_pct = 1
            min_inventory_pct = 0
            slope = (max_inventory_pct - min_inventory_pct) / (
                d_max_edge[symbol] - d_min_edge[symbol]
            )
            intercept = min_inventory_pct - slope * d_min_edge[symbol]

            inventory_pct = slope * abs(diff) + intercept
            inventory_pct = max(0, min(1, inventory_pct))
            if self.long_spread:
                max_buy = available_positions["PICNIC_BASKET"][0]
                max_sell = available_positions["UKULELE"][1]
                baseline = min(max_buy, abs(max_sell))
                if symbol == "BAGUETTE":
                    price = self.get_bid_price("BAGUETTE", state)
                    position = max(
                        available_positions["BAGUETTE"][1],
                        -baseline * inventory_pct * 2,
                    )
                elif symbol == "DIP":
                    price = self.get_bid_price("DIP", state)
                    position = max(
                        available_positions["DIP"][1], -baseline * inventory_pct * 4
                    )
                elif symbol == "UKULELE":
                    price = self.get_bid_price("UKULELE", state)
                    position = max(
                        available_positions["UKULELE"][1], -baseline * inventory_pct
                    )
                elif symbol == "PICNIC_BASKET":
                    price = self.get_ask_price("PICNIC_BASKET", state)
                    position = min(
                        available_positions["PICNIC_BASKET"][0],
                        baseline * inventory_pct,
                    )
            if self.short_spread:
                max_buy = available_positions["UKULELE"][0]
                max_sell = available_positions["PICNIC_BASKET"][1]
                baseline = min(max_buy, abs(max_sell))
                if symbol == "BAGUETTE":
                    price = self.get_ask_price("BAGUETTE", state)
                    position = min(
                        available_positions["BAGUETTE"][0],
                        -baseline * inventory_pct * 2,
                    )
                elif symbol == "DIP":
                    price = self.get_ask_price("DIP", state)
                    position = min(
                        available_positions["DIP"][0], -baseline * inventory_pct * 4
                    )
                elif symbol == "UKULELE":
                    price = self.get_ask_price("UKULELE", state)
                    position = min(
                        available_positions["UKULELE"][0], -baseline * inventory_pct
                    )
                elif symbol == "PICNIC_BASKET":
                    price = self.get_bid_price("PICNIC_BASKET", state)
                    position = max(
                        available_positions["PICNIC_BASKET"][1],
                        baseline * inventory_pct,
                    )
            return price, position

        if diff > 50:
            self.short_spread = True
            self.long_spread = False

        elif diff < -50:
            self.long_spread = True
            self.short_spread = False

        for symbol in symbols:
            if self.long_spread or self.short_spread:
                price, position = get_position(symbol, diff)
                orders[symbol].append(
                    Order(symbol, price, position)  # Create an Order object
                )

    def follow_alpha(self, state, orders):
        if self.position_filled:
            self.alpha = False
            self.position_filled = False
        symbols = [
            "PEARLS",
            "BANANAS",
            "COCONUTS",
            "PINA_COLADAS",
            "BERRIES",
            "DIVING_GEAR",
            "BAGUETTE",
            "DIP",
            "UKULELE",
            "PICNIC_BASKET",
        ]
        for symbol in symbols:
            if symbol in state.market_trades:
                buyer = state.market_trades[symbol][0].buyer
                seller = state.market_trades[symbol][0].seller
                quantity = state.market_trades[symbol][0].quantity

                if buyer == "Olivia" or seller == "Olivia":
                    self.alpha = True
                if self.alpha:
                    print("folloring Olivia ...")
                    if buyer == "Olivia":
                        max_buy, _ = self.get_available_position(symbol, state)
                        position_to_buy = min(quantity, max_buy)
                        orders[symbol].append(
                            Order(
                                symbol,
                                self.get_ask_price(symbol, state),
                                position_to_buy,
                            )
                        )
                        if position_to_buy == max_buy or position_to_buy == 0:
                            self.position_filled = True
                    elif seller == "Olivia":
                        _, max_sell = self.get_available_position(symbol, state)
                        position_to_sell = max(-quantity, max_sell)
                        orders[symbol].append(
                            Order(
                                symbol,
                                self.get_bid_price(symbol, state),
                                position_to_sell,
                            )
                        )
                        if position_to_sell == max_sell or position_to_sell == 0:
                            self.position_filled = True

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {
            "PEARLS": [],
            "BANANAS": [],
            "COCONUTS": [],
            "PINA_COLADAS": [],
            "BERRIES": [],
            "DIVING_GEAR": [],
            "BAGUETTE": [],
            "DIP": [],
            "UKULELE": [],
            "PICNIC_BASKET": [],
        }
        for symbol in state.order_depths:
            self.quote_bid_ask(symbol, state, orders)

        self.rv(state, orders)
        self.taking_strategy(state, orders)
        self.black_swan(state, orders)
        self.rv_2(state, orders)
        self.follow_alpha(state, orders)
        return orders
