import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pmdarima as pm
#from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm


class Plotter:
    
    @staticmethod
    def plot_prices_simple(df_data, str_product):
        df = df_data[df_data["product"] == str_product].copy()

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
    def plot_prices_full(df_data, str_product):
        df = df_data[df_data["product"] == str_product].copy()

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

class Regressor:

    @staticmethod
    def regress_simple(df_data, X_product, Y_product, element = "mid_price"):
        df = pd.DataFrame({
            X_product: df_data[df_data["product"] == X_product][element],
            Y_product: df_data[df_data["product"] == Y_product][element]
        })
        Regressor.linear_regression(df, [X_product], Y_product)

    @staticmethod
    def linear_regression(df_data, exog_labels: list, endog_label: str):
        df = df_data[exog_labels + [endog_label]]
        exog_string = ""
        for label in exog_labels:
            exog_string += f"+{label}"

        model = smf.ols(f"{endog_label} ~ {exog_string[1:]}",data = df)
        results = model.fit()
        print(results.summary())

    @staticmethod
    def quick_arima(df_data, product, element = "mid_price", seasonal = False, stepwise = False, suppress_warnings = True, print = True):
        df = df_data[df_data["product"] == product][element]

        model = pm.auto_arima(df.values, seasonal = seasonal, stepwise = stepwise, suppress_warnings = suppress_warnings, trace = print)
        order = model.order

        model = ARIMA(df.values, order = order).fit()
        return model, order
    
    @staticmethod
    def eval_arima(data, order, refit = False, training_split = .7, step = 10):
        train_index = int(len(data) * training_split)

        preds = []
        reals = []

        model = ARIMA(data[:train_index], order = order).fit()

        with tqdm(total = len(data) - train_index, desc = "Evaluating Arima") as pbar:
            while train_index < len(data):
                if refit: model = ARIMA(data[:train_index], order = order).fit()

                preds = preds + model.forecast(step).tolist()
                reals.append(data[train_index])

                pbar.update(step)

                train_index += step

        plt.plot(preds, label = "Arima Predictions")
        plt.plot(reals, label = "Real Values")

        plt.legend()
        plt.show()

    def arima_make_dataset(y, order):
        p, d, q = order

        # Step 1: Differencing
        y_diff = y.copy()
        for _ in range(d):
            y_diff = np.diff(y_diff)

        # Step 2: Create AR terms (lags of y_diff)
        X_ar = []
        y_targets = []

        for i in range(max(p, q), len(y_diff)):
            row = []

            # AR terms
            if p > 0:
                row += y_diff[i - p:i][::-1].tolist()

            # MA terms (approximate residuals)
            if q > 0:
                residuals = y_diff[i - q:i] - y_diff[i - q - 1:i - 1]
                row += residuals[::-1].tolist()

            X_ar.append(row)
            y_targets.append(y_diff[i])

        return np.array(X_ar), np.array(y_targets)
        # next diff

        return np.array(X), np.array(y_next)

    def forecast(y, model, steps=5, lag=5):
        dy = list(np.diff(y))
        preds = []

        for _ in range(steps):
            x_input = np.array(dy[-lag:]).reshape(1, -1)
            dy_pred = model.predict(x_input)[0]
            preds.append(dy_pred)
            dy.append(dy_pred)

        # Convert diffs back to full values
        return np.cumsum([y[-1]] + preds)[1:]
    
    def eval_model(y, model, steps=5, lag=5, training_split = .7):
        train_index = int(len(y) * training_split)

        preds = []
        reals = []

        with tqdm(total = len(y) - train_index, desc = "Evaluating Arima") as pbar:
            while train_index < len(y):

                preds = preds + Regressor.forecast(y[:train_index], model, steps, lag).tolist()
                reals = reals + y[train_index - steps: train_index].tolist()

                pbar.update(steps)

                train_index += steps

        plt.plot(reals, label = "Real Values")
        plt.plot(preds, label = "Arima Predictions")

        plt.legend()
        plt.show()

class ARIMA:
    def __init__(self, y, order):
        self.y = np.asarray(y, dtype=np.float64)
        self.p, self.d, self.q = order
        self.ar_coeffs = None
        self.ma_coeffs = None
        self.residuals = None
        self.last_value = None

    def difference(self, y, d):
        self.last_value = y.copy()
        for _ in range(d):
            y = np.diff(y)
        return y

    def inverse_difference(self, forecasted):
        last = self.last_value[-1]
        return np.r_[last, forecasted].cumsum()

    def fit(self, lr=1e-2, epochs=300):
        y_diff = self.difference(self.y, self.d) if self.d > 0 else y.copy()

        # Fit AR(p) using least squares
        if self.p > 0:
            Y = y_diff[self.p:]
            X = np.column_stack([y_diff[self.p - i - 1: -i - 1] for i in range(self.p)])
            self.ar_coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        else:
            self.ar_coeffs = np.array([])

        # Compute AR predictions and residuals
        preds = []
        for t in range(self.p, len(y_diff)):
            ar_part = sum(self.ar_coeffs[i] * y_diff[t - i - 1] for i in range(self.p))
            preds.append(ar_part)
        residuals = y_diff[self.p:] - np.array(preds)

        # Fit MA(q) using gradient descent
        if self.q > 0:
            ma_coeffs = np.zeros(self.q)

            for _ in range(epochs):
                grads = np.zeros_like(ma_coeffs)
                for t in range(len(residuals)):
                    ma_part = sum(ma_coeffs[i] * residuals[t - i - 1] if t - i - 1 >= 0 else 0 for i in range(self.q))
                    error = residuals[t] - ma_part
                    for i in range(self.q):
                        if t - i - 1 >= 0:
                            grads[i] += -2 * error * residuals[t - i - 1]
                ma_coeffs -= lr * grads / len(residuals)
            self.ma_coeffs = ma_coeffs
        else:
            self.ma_coeffs = np.array([])

        self.residuals = residuals.tolist()

        return self

    def forecast(self, steps):
        y_diff = self.difference(self.y, self.d) if self.d > 0 else y.copy()
        preds = []

        history = y_diff.tolist()
        residuals = self.residuals.copy()

        for _ in range(steps):
            ar_part = sum(self.ar_coeffs[i] * history[-i-1] for i in range(self.p)) if self.p > 0 else 0
            ma_part = sum(self.ma_coeffs[i] * residuals[-i-1] for i in range(self.q)) if self.q > 0 else 0
            forecast = ar_part + ma_part

            history.append(forecast)
            residuals.append(0)  # assume 0 residual in future
            preds.append(forecast)

        if self.d > 0:
            preds = self.inverse_difference(preds)

        return preds
