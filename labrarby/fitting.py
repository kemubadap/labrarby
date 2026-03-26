import numpy as np
from scipy.optimize import curve_fit
from .math_utils import create_model_function
from .base import BaseDataSet

class FittingMixin:
    """
    Mixin class providing methods for data fitting and Monte Carlo simulations.
    Expects self.data to be a numpy array.
    """

    def fit_function(self, x_col, y_col, function_expr, param_names, p0=None, sigma_col=None):
        """
        Fits a user-defined mathematical function to the data.
        
        Args:
            x_col, y_col (int): Column numbers for x and y coordinates (1-indexed).
            function_expr (str): Function expression (e.g., "a*x + b").
            param_names (list of str): List of parameter names (e.g., ["a", "b"]).
            p0 (list, optional): Initial parameter guesses. Defaults to 1s.
            sigma_col (int, optional): Column number for y uncertainties (1-indexed).
            
        Returns:
            tuple: (popt, sqrt_diag, R) 
                   Fitted parameters, standard deviations, and chi-squared statistic (if sigma provided).
        """
        if self.data is None:
            raise ValueError("No data available.")

        try:
            x = self.data[:, x_col - 1]
            y = self.data[:, y_col - 1]
        except IndexError:
            raise IndexError("Invalid x or y column index.")

        sigma = None
        if sigma_col is not None:
            try:
                sigma = self.data[:, sigma_col - 1]
            except IndexError:
                raise IndexError("Invalid uncertainty column index.")

        if p0 is None:
            p0 = [1] * len(param_names)
    
        # Używamy funkcji zaimportowanej z math_utils.py (bez self!)
        model = create_model_function(function_expr, param_names)
        
        try:
            if sigma is not None:
                popt, pcov = curve_fit(model, x, y, p0=p0, sigma=sigma, absolute_sigma=True)
            else:
                popt, pcov = curve_fit(model, x, y, p0=p0)
        except Exception as e:
            raise RuntimeError(f"Curve fitting failed: {e}")

        sqrt_diag = np.sqrt(np.diag(pcov))
        
        R = 0
        if sigma_col is not None and np.min(sigma) > 0:
            for xi, yi, si in zip(x, y, sigma):
                R += ((yi - model(xi, *popt)) / si) ** 2

        return popt, sqrt_diag, R

    def monte_carlo(self, x_col, y_col, xsigma_col, ysigma_col, model, n_params, N, p0=None):
        """
        Performs a Monte Carlo simulation for the given data and model function.
        
        Args:
            x_col, y_col (int): Data column numbers (1-indexed).
            xsigma_col, ysigma_col (int): Uncertainty column numbers for x and y (1-indexed).
            model (callable): Model function, e.g., model(x, a, b) = a*x + b.
            n_params (int): Number of parameters in the model function.
            N (int): Number of Monte Carlo simulations.
            p0 (list, optional): Initial parameter guesses.

        Returns:
            tuple: (popt, sqrt_diag, R) Mean parameters, standard deviations, and chi-squared statistic.
        """
        if self.data is None:
            raise ValueError("No data available.")

        try:
            x = self.data[:, x_col - 1]
            y = self.data[:, y_col - 1]
            xsigma = self.data[:, xsigma_col - 1]
            ysigma = self.data[:, ysigma_col - 1]
        except IndexError:
            raise IndexError("Invalid column index provided.")

        popt_values = np.zeros((N, n_params))

        for i in range(N):
            x_sim = np.random.normal(x, xsigma)
            y_sim = np.random.normal(y, ysigma)
            try:
                popt, _ = curve_fit(model, x_sim, y_sim, p0)
                popt_values[i] = popt
            except Exception as e:
                print(f"Warning: Fit failed in iteration {i}: {e}")
                popt_values[i] = np.nan

        popt = np.nanmean(popt_values, axis=0)
        sqrt_diag = np.nanstd(popt_values, axis=0)

        R = 0
        if np.min(ysigma) > 0:
            for xi, yi, si in zip(x, y, ysigma):
                R += ((yi - model(xi, *popt)) / si) ** 2

        return popt, sqrt_diag, R

    def naive_linear_fit(self, x_col, y_col, xsigma_col, ysigma_col, naive_b=False):
        """
        Fits extreme lines to data using a naive method based on uncertainty rectangles.
        
        Args:
            x_col, y_col (int): Data column numbers (1-indexed).
            xsigma_col, ysigma_col (int): Uncertainty column numbers (1-indexed).
            naive_b (bool, optional): If True, the averaged b is calculated only based 
                                      on the two extreme points. Defaults to False.
          
        Returns:
            list: Three 2-element lists: [[a_max, b_max], [a_min, b_min], [a_avg, b_avg]]
        """
        if self.data is None:
            raise ValueError("No data available.")

        try:
            x = self.data[:, x_col - 1]
            y = self.data[:, y_col - 1]
            dx = self.data[:, xsigma_col - 1]
            dy = self.data[:, ysigma_col - 1]
        except IndexError:
            raise IndexError("Invalid column indices provided.")

        idx_min = np.argmin(x)
        idx_max = np.argmax(x)

        if idx_min == idx_max:
            raise ValueError("Extreme points are identical (no span on the X-axis).")

        x1, y1, dx1, dy1 = x[idx_min], y[idx_min], dx[idx_min], dy[idx_min]
        x2, y2, dx2, dy2 = x[idx_max], y[idx_max], dx[idx_max], dy[idx_max]

        corners1 = [
            (x1 + dx1, y1 + dy1), (x1 + dx1, y1 - dy1),
            (x1 - dx1, y1 + dy1), (x1 - dx1, y1 - dy1)
        ]
        corners2 = [
            (x2 + dx2, y2 + dy2), (x2 + dx2, y2 - dy2),
            (x2 - dx2, y2 + dy2), (x2 - dx2, y2 - dy2)
        ]

        a_max, b_max = -np.inf, 0
        a_min, b_min = np.inf, 0

        for cx1, cy1 in corners1:
            for cx2, cy2 in corners2:
                if cx1 == cx2:
                    continue
                
                a = (cy2 - cy1) / (cx2 - cx1)
                b = cy1 - a * cx1
                
                if a > a_max:
                    a_max = a
                    b_max = b
                if a < a_min:
                    a_min = a
                    b_min = b

        a_avg = (a_max + a_min) / 2.0

        if naive_b:
            b1 = y1 - a_avg * x1
            b2 = y2 - a_avg * x2
            b_avg = (b1 + b2) / 2.0
        else:
            def model_fixed_a(x_val, b_val):
                return a_avg * x_val + b_val

            try:
                if np.all(dy > 0):
                    popt, _ = curve_fit(model_fixed_a, x, y, sigma=dy, absolute_sigma=True)
                else:
                    popt, _ = curve_fit(model_fixed_a, x, y)
                    
                b_avg = popt[0]
            except Exception as e:
                raise RuntimeError(f"Error fitting averaged b: {e}")

        return [[a_max, b_max], [a_min, b_min], [a_avg, b_avg]]

class FittingDataSet(BaseDataSet, FittingMixin):
    """
    A DataSet class that includes fitting capabilities.
    """
    pass