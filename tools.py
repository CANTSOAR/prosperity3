import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.robust.robust_linear_model as rlm
import statsmodels.api as sm

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

class Regression:
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
            Regression.plot_residuals(df_reg, results)
        
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