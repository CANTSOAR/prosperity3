import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    
    @staticmethod
    def plot_prices_simple(df_prices, str_product):
        df = df_prices[df_prices["product"] == str_product].copy()

        df.index = pd.to_numeric(df.index)
        df = df.sort_index()

        df["volume"] = df["bid_volume_1"].fillna(0) + df["ask_volume_1"].fillna(0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.scatter(df.index, df["mid_price"], color="blue", s=10)
        ax1.set_title(f"{str_product} Price Chart", fontsize=14)
        ax1.set_ylabel("Mid Price")

        ax2.scatter(df.index, df["volume"], color="gray")
        ax2.set_title(f"{str_product} Volume", fontsize=12)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Timestamp")

        step = 1000
        ticks = df.index[::step]
        ax2.set_xticks(ticks)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_prices_full(df_prices, str_product):
        df = df_prices[df_prices["product"] == str_product].copy()

        df.index = pd.to_numeric(df.index)
        df = df.sort_index()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        for i in range(1, 4):
            max_vol = max(df[f"bid_volume_{i}"].tolist() + df[f"ask_volume_{i}"].tolist())
            ax1.scatter(df.index, df[f"bid_price_{i}"], color=[(0, 1, 0, a / max_vol) for a in df[f"bid_volume_{i}"]], s=10)
            ax1.scatter(df.index, df[f"ask_price_{i}"], color=[(1, 0, 0, a / max_vol) for a in df[f"ask_volume_{i}"]], s=10)

            ax2.scatter(df.index, df[f"bid_volume_{i}"], color=[1 if j == i else 0 for j in range(1, 4)], label = ["howdy", "L1", "L2", "L3"][i])
            ax2.scatter(df.index, df[f"ask_volume_{i}"], color=[1 if j == i else 0 for j in range(1, 4)])
        
        ax1.set_title(f"{str_product} Price Chart", fontsize=14)
        ax1.set_ylabel("Mid Price")
        ax2.set_title(f"{str_product} Volume", fontsize=12)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Timestamp")
        ax2.legend()

        step = 1000
        ticks = df.index[::step]
        ax2.set_xticks(ticks)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()