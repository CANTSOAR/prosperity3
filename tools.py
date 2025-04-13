import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pmdarima as pm
#from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

import statsmodels.robust.robust_linear_model as rlm
import statsmodels.api as sm

import os
import json
from io import StringIO
import re

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
        Regressor.regression(df, [X_product], Y_product)
    """
    A class for performing different types of regression analysis on financial time series data.
    """

    @staticmethod
    def regression(df, x, y, element="mid_price", robust=True, plot=False):
        # Extract the time series data
        X = df[df["product"]==x][element]
        Y = df[df["product"]==y][element]
        
        # Create dataframe with input and output
        df_reg = pd.DataFrame({
            "input": X.reset_index(drop=True), 
            "output": Y.reset_index(drop=True)
        })
        
        # Initial regression - choose between robust and OLS
        if robust:
            model = smf.rlm("output ~ input", data=df_reg)
        else:
            model = smf.ols("output ~ input", data=df_reg)
            
        results = model.fit()
        
        # Add residuals to the dataframe
        df_reg["residuals"] = results.resid
        
        # Plot the original data and regression line if requested
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_reg["input"], df_reg["output"], alpha=0.5, color='blue', label='Data Points')
            
            # Generate predictions for the regression line
            input_range = np.linspace(df_reg["input"].min(), df_reg["input"].max(), 100)
            if robust:
                pred = results.params[0] + results.params[1] * input_range
            else:
                pred = results.predict(pd.DataFrame({"input": input_range}))
                
            ax.plot(input_range, pred, 'r-', linewidth=2, label='Regression Line')
            
            ax.set_title(f'Regression: {y} vs {x} ({element})')
            ax.set_xlabel(f'{x} {element}')
            ax.set_ylabel(f'{y} {element}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Plot residuals
            Regressor.plot_residuals(df_reg, results)
        
        return results, df_reg

    @staticmethod
    def residual_regression(df_reg, results, lags=1, robust=True, plot=False):
        for lag in range(1, lags+1):
            df_reg[f"residuals_lag_{lag}"] = df_reg["residuals"].shift(lag)
        
        # Drop NaN values that result from shifting
        df_reg_lagged = df_reg.dropna()
        
        # Define the formula based on the number of lags
        lag_vars = " + ".join([f"residuals_lag_{i}" for i in range(1, lags+1)])
        formula = f"residuals ~ {lag_vars}"
        
        # Regress residuals on lagged residuals
        if robust:
            resid_model = smf.rlm(formula, data=df_reg_lagged)
        else:
            resid_model = smf.ols(formula, data=df_reg_lagged)
            
        resid_results = resid_model.fit()
        
        if plot:
            # Create additional plots specific to residual regression
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Actual vs Predicted residuals
            axes[0].scatter(df_reg_lagged["residuals"], resid_results.fittedvalues, alpha=0.5)
            axes[0].plot([df_reg_lagged["residuals"].min(), df_reg_lagged["residuals"].max()], 
                       [df_reg_lagged["residuals"].min(), df_reg_lagged["residuals"].max()], 
                       'r--', linewidth=1)
            axes[0].set_title('Actual vs Predicted Residuals')
            axes[0].set_xlabel('Actual Residuals')
            axes[0].set_ylabel('Predicted Residuals')
            axes[0].grid(True, alpha=0.3)
            
            # Residuals of residuals
            resid_of_resid = resid_results.resid
            axes[1].hist(resid_of_resid, bins=20, alpha=0.7)
            axes[1].set_title('Histogram of Residuals of Residual Regression')
            axes[1].set_xlabel('Residual Value')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # ACF plot for residuals of residuals
            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                sm.graphics.tsa.plot_acf(resid_of_resid, lags=min(20, len(resid_of_resid) // 2), ax=ax)
                ax.set_title('ACF of Residuals of Residual Regression')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not plot ACF: {e}")
        
        return resid_results

    @staticmethod
    def plot_residuals(df_reg, results):
        """
        Plot comprehensive residual diagnostics.
        
        Parameters:
        -----------
        df_reg : pandas.DataFrame
            DataFrame containing the regression data and residuals
        results : statsmodels Results object
            Results from the regression
        """
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Plot residuals over time
        axes[0, 0].plot(df_reg.index, df_reg["residuals"], 'o-', alpha=0.5, markersize=3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time Index')
        axes[0, 0].set_ylabel('Residual Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals histogram with normal distribution overlay
        resid = df_reg["residuals"].dropna()
        axes[0, 1].hist(resid, bins=30, density=True, alpha=0.7)
        
        # Add normal distribution for comparison
        xmin, xmax = axes[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        mean, std = resid.mean(), resid.std()
        p = np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
        axes[0, 1].plot(x, p, 'r--', linewidth=2)
        
        # Add mean and std annotations
        axes[0, 1].annotate(f'Mean: {mean:.2f}\nStd: {std:.2f}', 
                          xy=(0.7, 0.85), xycoords='axes fraction',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        axes[0, 1].set_title('Histogram of Residuals with Normal Distribution')
        axes[0, 1].set_xlabel('Residual Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Lag plot to check for autocorrelation
        pd.plotting.lag_plot(df_reg["residuals"], lag=1, ax=axes[1, 0])
        axes[1, 0].set_title('Lag Plot of Residuals (lag=1)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. QQ plot to check normality
        sm.qqplot(resid, line='45', ax=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot for autocorrelation
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            lags = min(20, len(df_reg["residuals"]) // 2)
            sm.graphics.tsa.plot_acf(df_reg["residuals"].dropna(), lags=lags, ax=ax)
            ax.set_title('Autocorrelation Function (ACF) of Residuals')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Could not plot ACF: {e}")
        plt.tight_layout()
        plt.show()
        
        # Plot the relationship between fitted values and residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(results.fittedvalues, results.resid, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Residuals vs Fitted Values')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.grid(True, alpha=0.3)
        
        # Add a lowess smooth line
        try:
            lowess = sm.nonparametric.lowess(results.resid, results.fittedvalues, frac=0.3)
            ax.plot(lowess[:, 0], lowess[:, 1], 'r-', linewidth=2)
        except Exception as e:
            print(f"Could not add lowess smoothing: {e}")
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def quick_arima(df_data, product, element = "mid_price", seasonal = False, stepwise = False, suppress_warnings = True, print = True, need_model = True):
        df = df_data[df_data["product"] == product][element]

        model = pm.auto_arima(df.values, seasonal = seasonal, stepwise = stepwise, suppress_warnings = suppress_warnings, trace = print)
        order = model.order

        if need_model: model = ARIMA(df.values, order = order).fit()
        else: model = None

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

    @staticmethod
    def arima_make_dataset(y, order):
        p, d, q = order

        # Step 1: Differencing
        y_diff = y.copy()
        for _ in range(d):
            y_diff = np.diff(y_diff)

        X_ar = []
        y_targets = []

        for i in range(max(p, q + 1), len(y_diff)):
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

    @staticmethod
    def forecast(y, model, order, steps):
        p, d, q = order

        # Apply differencing
        y_diff = y.copy()
        for _ in range(d):
            y_diff = np.diff(y_diff)
        dy = list(y_diff)

        preds = []

        for _ in range(steps):
            row = []

            if p > 0:
                row += dy[-p:][::-1]

            if q > 0:
                residuals = np.array(dy[-q:]) - np.array(dy[-q-1:-1])
                row += residuals[::-1].tolist()

            x_input = np.array(row).reshape(1, -1)
            dy_pred = model.predict(x_input)[0]
            preds.append(dy_pred)
            dy.append(dy_pred)

        # Inverse differencing to get actual y values
        return np.cumsum([y[-1]] + preds)[1:]

    @staticmethod
    def eval_model(y, model, order, steps, training_split = .7):
        p, d, q = order
        train_index = int(len(y) * training_split)

        preds = []
        reals = []

        with tqdm(total=len(y) - train_index, desc="Evaluating ARIMA") as pbar:
            while train_index + steps <= len(y):
                y_train = y[:train_index]
                y_real = y[train_index:train_index + steps]

                y_pred = Regressor.forecast(y_train, model, steps=steps, order=order)
                preds.extend(y_pred)
                reals.extend(y_real)

                train_index += steps
                pbar.update(steps)

        plt.plot(reals, label="Real Values")
        plt.plot(preds, label="ARIMA Predictions")
        plt.legend()
        plt.title(f"ARIMA{order} Model Evaluation")
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
        y_diff = self.difference(self.y, self.d) if self.d > 0 else self.y.copy()

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
        y_diff = self.difference(self.y, self.d) if self.d > 0 else self.y.copy()
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
    

import os
import json
import re
import pandas as pd
from io import StringIO
import plotly.graph_objects as go

class Log_Analysis:

    def analyse(self, product, file_index=-1, backtest_or_submission=True):
        folder = "backtests" if backtest_or_submission else "submissions"
        full_path = os.path.relpath(folder)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Folder {full_path} does not exist")

        files = sorted([
            os.path.join(full_path, f)
            for f in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, f))
        ])

        file = files[file_index]
        sandbox_lines = []
        activity_lines = []
        trade_lines = []

        with open(file, "r", encoding="utf-8") as f:
            sandbox_section = False
            activity_section = False
            trade_section = False

            for line in f:
                line = line.strip()

                if line == "Sandbox logs:":
                    sandbox_section = True
                    continue

                if line == "Activities log:":
                    activity_section = True
                    sandbox_section = False
                    continue

                if line == "Trade History:":
                    activity_section = False
                    trade_section = True
                    continue

                if sandbox_section:
                    sandbox_lines.append(line)
                elif activity_section:
                    activity_lines.append(line)
                elif trade_section:
                    trade_lines.append(line)

        # Parse sandbox logs
        sandbox_json = self.parse_multiple_json_objects(sandbox_lines)

        lambda_logs = {
            entry["timestamp"]: json.loads(entry["lambdaLog"])
            for entry in sandbox_json
        }

        # Parse activities
        activity_content = "\n".join(activity_lines)
        prices = pd.read_csv(StringIO(activity_content), sep=";")
        prices.index = prices["timestamp"]

        # Parse trades
        trades_content = "\n".join(trade_lines)
        trades_content = re.sub(r',\s*(\}|\])', r'\1', trades_content)
        trade_json = json.loads(trades_content)
        trades = pd.DataFrame(trade_json)
        trades["timestamp"] = trades["timestamp"].astype(int)
        trades.index = trades["timestamp"]

        # Filter by product
        product_prices = prices[prices["product"] == product]
        trades = trades[trades["symbol"] == product]
        our_buys = trades[trades["buyer"] == "SUBMISSION"]
        our_sells = trades[trades["seller"] == "SUBMISSION"]

        # Simplified hover text
        def summarize_log(log):
            return log[-1]

        product_prices["hover_text"] = product_prices["timestamp"].map(
            lambda ts: summarize_log(lambda_logs.get(ts, []))
        )

        trades["hover_text"] = trades["timestamp"].map(
            lambda ts: summarize_log(lambda_logs.get(ts, []))
        )

        # Plot using Plotly
        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=product_prices.index,
            y=product_prices["mid_price"],
            mode="lines",
            name="Prices",
            text=product_prices["hover_text"],
            hoverinfo="text",
            line=dict(color="black")
        ))

        # All trades
        fig.add_trace(go.Scatter(
            x=trades.index,
            y=trades["price"],
            mode="markers",
            name="All Trades",
            text=trades["hover_text"],
            hoverinfo="text",
            marker=dict(color="blue")
        ))

        # Our buys
        fig.add_trace(go.Scatter(
            x=our_buys.index,
            y=our_buys["price"],
            mode="markers",
            name="Our Buys",
            text=our_buys["timestamp"].map(lambda_logs).map(summarize_log),
            hoverinfo="text",
            marker=dict(color="green", symbol="circle")
        ))

        # Our sells
        fig.add_trace(go.Scatter(
            x=our_sells.index,
            y=our_sells["price"],
            mode="markers",
            name="Our Sells",
            text=our_sells["timestamp"].map(lambda_logs).map(summarize_log),
            hoverinfo="text",
            marker=dict(color="red", symbol="x")
        ))

        fig.update_layout(title=f"{product} Price and Trades with Hover Logs",
                          xaxis_title="Timestamp",
                          yaxis_title="Price",
                          hovermode="closest")

        fig.write_html(
            "plots/plot_output.html",
            auto_open=True,
            config={"scrollZoom": True, "displaylogo": False}
        )

    def parse_multiple_json_objects(self, lines):
        objects = []
        buffer = []
        depth = 0
        in_object = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if not in_object and "{" in stripped:
                in_object = True

            if in_object:
                buffer.append(line)
                depth += line.count("{")
                depth -= line.count("}")

                if depth == 0:
                    try:
                        joined = "\n".join(buffer)
                        obj = json.loads(joined)
                        objects.append(obj)
                    except json.JSONDecodeError as e:
                        print(f"[!] Failed to parse JSON object:\n{joined[:200]}...\nError: {e}")
                    buffer = []
                    in_object = False

        return objects



pray = Log_Analysis()
pray.analyse("PICNIC_BASKET1")