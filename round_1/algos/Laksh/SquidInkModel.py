from datamodel import Order, TradingState, OrderDepth
from typing import List
import numpy as np
import jsonpickle

# ====== Begin: Paste your trained model weights and normalization parameters ======

# Model architecture:
#   FC1: Linear layer with weight matrix W1 (shape [32, 6]) and bias b1 (shape [32])
#   FC2: Linear layer with weight matrix W2 (shape [16, 32]) and bias b2 (shape [16])
#   FC3: Linear layer with weight matrix W3 (shape [1, 16]) and bias b3 (shape [1])
# (These are example placeholders; replace the [...] with your actual trained model parameters.)

W1 = np.array([
    [-0.2838,  0.2987,  0.4890,  0.6750, -0.0515,  0.4325],
    [ 0.0240,  0.1613, -0.7709, -0.8341,  0.3646,  0.5710],
    [ 0.7047,  0.5989,  0.0696, -0.0587,  0.1155, -0.4615],
    [-0.0806, -0.0330,  0.5145, -0.7049, -0.0978, -0.4485],
    [-0.0138, -0.3343,  0.0657,  0.8485, -0.7689,  0.1318],
    [ 0.0743,  0.0285,  0.0691, -0.6221, -0.7045,  0.3744],
    [-0.0732, -1.0735,  0.5735, -0.6036,  0.1445,  0.3758],
    [-0.3896, -0.2533, -0.3347, -0.1989, -0.3096, -0.1558],
    [-0.3153,  0.3679, -0.5048, -0.2447, -0.5457,  0.2218],
    [ 0.4961, -0.4705, -0.1785,  0.1199,  0.4257, -0.2181],
    [ 0.3885, -0.5138, -0.5446, -0.3965,  0.1471,  0.4166],
    [ 0.0370, -0.0367,  0.2457,  0.4407, -0.5044, -0.6844],
    [-0.6226, -0.0844, -0.5856,  0.0144, -0.5914, -0.1862],
    [-0.7157, -0.0929, -0.1678,  0.5717,  0.4988,  0.0414],
    [-0.2704,  0.9273, -1.0146, -0.1959,  0.2975,  0.2138],
    [ 0.3624, -0.6003, -0.3137,  0.5572,  0.0838,  0.1097],
    [ 0.1196, -0.3995, -0.6890,  0.1522,  0.9252,  0.4962],
    [-0.1057,  0.0126,  0.8986, -0.5741,  0.0563, -0.8669],
    [-0.4954, -0.4626, -0.3356, -0.0885,  0.5099,  0.4142],
    [-0.2065, -0.2354,  0.5378,  0.3429,  0.4247, -0.1659],
    [ 0.3606,  0.2439,  0.4135,  0.1575, -0.6170, -0.3323],
    [-0.1872,  0.3611, -0.3444, -0.8160,  0.9500, -0.0134],
    [-0.5049, -0.4710,  0.5030,  0.6563, -0.2075, -0.1511],
    [-0.5685,  0.5801, -0.2848, -0.0137,  0.3591, -0.0802],
    [ 0.1810, -0.2198, -0.0427, -0.5208,  0.6186,  0.9803],
    [ 0.2832,  0.3096,  0.2389, -0.6233, -0.1466, -0.6041],
    [ 0.6071,  0.7347,  0.7428, -0.5488, -0.6417, -0.6003],
    [-0.1581,  0.0406,  0.0512,  0.5388,  0.4889, -0.8636],
    [ 0.6133, -0.0645, -0.6369,  0.3670, -0.1475, -0.2279],
    [-0.0657, -0.3276, -0.7950,  0.2493,  0.1490,  0.1251],
    [-0.1360,  0.4226,  0.4087,  0.5222, -0.3218, -0.3085],
    [ 0.1850, -0.8476, -0.7815,  0.5051,  0.6253,  0.7397]
], dtype=np.float32)

b1 = np.array([
    # Paste your 32 biases for FC1 here.
    -0.1386,  0.0701, -0.1436, -0.5276,  0.1122,  0.3888,  0.2210, -0.7456,
    0.1122,  0.9161,  0.0744,  0.6568, -0.2512,  0.2594,  0.2346,  0.0659,
    0.3220,  0.4396,  0.5456,  0.4464,  0.0847,  0.0094,  0.0023,  0.4011,
    -0.4263,  0.9371, -0.0556, -0.0515,  0.1257, -0.7779,  0.1878, -0.0623
], dtype=np.float32)

W2 = np.array([
    # Paste your 16x32 weight matrix for FC2 here.
    # For example purposes only, a few sample rows are shown:
    [ 0.2709, -1.2104,  0.4893, -0.2149,  0.5861,  0.0231, -0.1696, -0.1374,
        -0.1684,  0.1603, -0.2237,  0.1821,  0.2714, -0.8478, -0.0928, -0.1868,
        -0.0111,  0.5155, -0.8597, -0.0388,  0.0906, -0.5604, -0.1853,  0.6062,
        0.0391,  0.0066, -0.2798, -0.0064, -0.0594,  0.2476,  0.1649, -0.3864],
    [ 0.0569, -0.2025,  0.3001, -0.1907, -0.2328, -0.4886,  0.3125,  0.1370,
        0.0697, -0.4509, -0.4834, -0.1766, -0.2085,  0.2557,  0.4840,  0.3557,
        -0.3277,  0.0293,  0.0493,  0.0370,  0.1379, -0.2596,  0.2078, -0.1599,
        -0.2327,  0.1802,  0.7152, -0.4133,  0.0766, -0.0534,  0.3892,  0.5664],
    [-0.0769,  0.7122, -0.1866, -0.0725,  0.3901,  0.0829, -0.2081,  0.5370,
        -0.0622,  0.2725, -0.4435, -0.2359, -0.0184, -0.2901, -0.5950, -0.5956,
        -0.1711,  0.3597,  0.4551, -0.6511,  0.1314,  0.5431,  0.0167, -0.4876,
        0.1797, -0.1555,  0.6188, -0.8966, -0.3575, -0.4903,  0.3965,  0.5353],
    [ 0.1829,  0.1076, -0.1003, -0.1701, -0.0319,  0.2414,  0.0231,  0.0569,
        -0.2033, -0.2051,  0.3138, -0.4928,  0.0756,  0.1387, -0.0567,  0.3021,
        0.2714,  0.2957,  0.2819, -0.3571, -0.4392, -0.1203,  0.1694,  0.1257,
        -0.2197,  0.0459, -0.0671,  0.0293,  0.1738,  0.4529, -0.1315,  0.1367],
    [ 0.0643, -0.1041,  0.0756, -0.5575, -0.1375,  0.2749,  0.1212,  0.1888,
        0.1717,  0.2245, -0.1510,  0.1321,  0.0844, -0.4395,  0.5437, -0.0316,
        0.1338, -0.1926,  0.0982, -0.0160, -0.1881, -0.2015,  0.0391, -0.6748,
        -0.4339,  0.1945, -0.0850, -0.0688,  0.1935,  0.1681,  0.1342,  0.5191],
    [ 0.0633,  0.0292, -0.3722,  0.0159,  0.0915,  0.2342, -0.1844,  0.3005,
        0.0664,  0.1604,  0.1603,  0.2799, -0.3693,  0.3659, -0.5040,  0.0501,
        0.3636, -0.2198,  0.1008, -0.3516,  0.1636,  0.6470, -0.3353, -0.1573,
        0.4497, -0.2750, -0.4402,  0.0651, -0.0327,  0.4333,  0.3367,  0.4314],
    [-0.2115,  0.3313, -0.4402, -0.3902, -0.0946, -0.5639,  0.3905,  0.7201,
        -0.0776, -0.2365, -0.2512, -0.5533, -0.3859,  0.2441,  0.3406,  0.1523,
        0.2537,  0.7463,  0.3997, -0.5057,  0.0078, -0.2923,  0.3609,  0.5521,
        -0.0473, -0.0264,  0.1414,  0.0757,  0.6158,  0.5996, -0.6020, -0.1681],
    [ 0.1696, -0.1287,  0.1242, -0.0832,  0.4369,  0.0396, -0.0775, -0.6095,
        0.1617,  0.0445, -0.1505, -0.2890,  0.0013,  0.1688, -0.0909,  0.4120,
        -0.3236, -0.1847, -0.0108,  0.1377, -0.5915, -0.1853,  0.3061, -0.0074,
        0.0272,  0.6691, -0.3913,  0.1063, -0.4874, -0.4801, -0.0838,  0.3581],
    [ 0.2468, -0.1765,  0.1601, -0.1505,  0.0570,  0.1752,  0.2084,  0.1793,
        0.1255, -0.0219, -0.3070,  0.0220,  0.0966,  0.1450,  0.3503, -0.0840,
        -0.3476, -0.1055, -0.2758,  0.0556,  0.1766, -0.0980,  0.1248, -0.1492,
        -0.4730,  0.1603,  0.1713,  0.3514,  0.0688,  0.2336,  0.0917,  0.0763],
    [-0.0062, -0.0576,  0.1903,  0.3976,  0.0885,  0.1319,  0.1624,  0.1764,
        0.1439,  0.3167, -0.1133,  0.1063,  0.0939, -0.0827,  0.1694, -0.2035,
        -0.3829,  0.3155,  0.1245,  0.1611,  0.1005,  0.7102,  0.0749, -0.2273,
        0.1418,  0.1484,  0.0375,  0.3847, -0.0275, -0.0335, -0.0995, -0.1216],
    [ 0.7239, -0.1754, -0.3462, -0.4686,  0.2524,  0.0646,  0.0832, -0.1588,
        0.1614, -0.0908,  0.2850,  0.2078, -0.2831, -0.1393, -0.0979, -0.2875,
        0.3129,  0.1028,  0.2569, -0.3143,  0.0268, -0.0502, -0.4432,  0.1554,
        0.0814,  0.1377, -0.1763,  0.0784, -0.0447, -0.1209,  0.0093, -0.1395],
    [-0.0712, -0.7660, -0.0474, -0.2321,  0.6012,  0.4407, -0.1356, -0.6462,
        -0.3216, -0.0422, -0.3877,  0.3645, -0.0425, -0.1709,  0.1894, -0.2553,
        -0.2850,  0.4455, -0.6679,  0.1799, -0.2695, -0.2168, -0.4070,  0.3709,
        0.6253,  0.2046, -0.2461,  0.1815, -0.2859,  0.5667,  0.2581, -0.6230],
    [ 0.3812, -0.1514,  0.1898, -0.4283, -0.0346,  0.0840,  0.2462, -0.0622,
        0.0090, -0.1250, -0.8354, -0.0296,  0.1204, -0.0061,  0.2898,  0.4506,
        -0.3817, -0.1866, -0.3175,  0.0457, -0.0908,  0.0778,  0.4417, -0.1947,
        -0.3816,  0.2590,  0.3330,  0.2559, -0.1688, -0.0443,  0.2384,  0.2265],
    [-0.2430, -0.2499, -0.0865,  0.0352, -0.3497, -0.3018,  0.7450,  0.7250,
        -0.7212,  0.2560, -0.6795, -0.4914,  0.0494,  0.3197,  0.3092,  0.6950,
        -0.5207, -0.2207, -0.3955, -0.0258, -0.6093, -0.0960,  0.1383,  0.2908,
        -0.3167, -0.1354,  0.5850, -0.1453, -0.1320, -0.0677, -0.4370,  0.4226],
    [-0.2213, -0.4346,  0.1827, -0.4949, -0.0481,  0.4168, -0.0478,  0.0771,
        -0.3533,  0.2152,  0.2158, -0.0157,  0.3011, -0.2888,  0.0171, -0.1079,
        -0.3939, -0.1853,  0.0845, -0.0709, -0.1913,  0.3280,  0.1675,  0.1175,
        -0.4724,  0.1756, -0.0301, -0.1037,  0.1105, -0.0223, -0.2453,  0.3589],
    [-0.1979, -0.1834,  0.1863, -0.1015, -0.2300, -0.3501,  0.4184, -0.0740,
        0.4813, -0.3277, -0.3729, -0.2994, -0.2107, -0.5129,  0.2201, -0.0809,
        -0.5846, -0.1662, -0.2204, -0.0401, -0.6861,  0.3107, -0.8672, -0.3784,
        0.1483,  0.5090,  0.3034, -0.3384,  0.3735, -0.3627, -0.0897,  0.8835]
], dtype=np.float32)

b2 = np.array([
    # Paste your 16 biases for FC2 here.
    -0.2919,  0.1632, -0.1567, -0.0080,  0.5240,  0.1583, -0.1573, -0.0746,
    0.0145,  0.0209,  0.0881, -0.5409,  0.1578,  0.2029,  0.1803, -0.1500
], dtype=np.float32)

W3 = np.array([
    # Paste your 1x16 weight matrix for FC3 here.
    [-0.6233,  0.7539, -0.8306, -0.3716, -0.3539,  0.6820, -0.5592, -0.6877,
    0.2976,  0.3431, -0.4988, -0.6373,  0.3283, -0.9333, -0.3952, -0.9162]
], dtype=np.float32)

b3 = np.array([0.4145], dtype=np.float32)

# Normalization parameters (from training)
X_mean = np.array([1971.11832366, 1971.11273921, 1971.11287257, 1971.11297259, 1971.11292258, 1971.11298926], dtype=np.float32)
X_std  = np.array([67.8927424, 67.89680279, 67.89680355, 67.89684832, 67.89682354, 67.89682343], dtype=np.float32)
y_mean = 1971.1076381943055
y_std  = 67.90087110630998

# ====== End: Paste your trained model weights and normalization parameters ======

class Trader:
    def run(self, state: TradingState):
        # Load persistent trader state from previous iterations.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}

        # Initialize persistent data for SQUID_INK price history if not present.
        if "squid_ml" not in trader_state:
            trader_state["squid_ml"] = {"price_history": []}
        ml_state = trader_state["squid_ml"]

        result = {}  # Dictionary for orders

        # Process only SQUID_INK.
        for product in state.order_depths:
            if product != "SQUID_INK":
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Determine current mid price from the order book.
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                current_mid = (best_bid + best_ask) / 2
            else:
                current_mid = 1971.0  # Fallback if order book is incomplete.
            print("SQUID_INK - Current mid price:", current_mid)

            # Update rolling price history (store last 6 mid prices).
            ml_state["price_history"].append(current_mid)
            if len(ml_state["price_history"]) > 6:
                ml_state["price_history"] = ml_state["price_history"][-6:]

            # Only proceed with prediction if we have 6 values.
            if len(ml_state["price_history"]) < 6:
                print("SQUID_INK - Not enough price history for prediction. Skipping trade.")
            else:
                # Get the input vector (last 6 mid prices).
                x = np.array(ml_state["price_history"], dtype=np.float32)  # shape (6,)
                # Normalize the input.
                x_norm = (x - X_mean) / X_std
                # Forward pass through the neural network.

                # FC1: z1 = W1 * x_norm + b1
                # Note: W1 is shape (32,6); x_norm shape (6,) so z1 will have shape (32,)
                z1 = np.dot(W1, x_norm) + b1
                # ReLU activation
                a1 = np.maximum(0, z1)

                # FC2: z2 = W2 * a1 + b2  (W2: shape (16,32); a1: shape (32,) -> (16,))
                z2 = np.dot(W2, a1) + b2
                a2 = np.maximum(0, z2)

                # FC3: z3 = W3 * a2 + b3  (W3: shape (1,16); a2: shape (16,) -> (1,))
                z3 = np.dot(W3, a2) + b3
                # Predicted normalized output.
                y_norm_pred = z3[0]
                # Denormalize the prediction.
                predicted_mid = y_norm_pred * y_std + y_mean
                print("SQUID_INK - Predicted next mid price:", predicted_mid)

                # Generate trading signal based on the difference.
                threshold = 1.0  # Adjust threshold as needed.
                if predicted_mid > current_mid + threshold:
                    # Buy signal: place a BUY order at the current best ask.
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        print("SQUID_INK - Buy signal. Placing BUY order for 1 unit at", best_ask)
                        orders.append(Order(product, best_ask, 1))
                    else:
                        print("SQUID_INK - Buy signal but no sell orders available.")
                elif predicted_mid < current_mid - threshold:
                    # Sell signal: place a SELL order at the current best bid.
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        print("SQUID_INK - Sell signal. Placing SELL order for 1 unit at", best_bid)
                        orders.append(Order(product, best_bid, -1))
                    else:
                        print("SQUID_INK - Sell signal but no buy orders available.")
                else:
                    print("SQUID_INK - Prediction within threshold. No trade executed.")

            result[product] = orders

        # Update persistent state.
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion operations.
        return result, conversions, traderData
