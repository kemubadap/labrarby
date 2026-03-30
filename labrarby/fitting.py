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

    def naive_linear_fit(self, x_col, y_col, xsigma_col, ysigma_col, naive_b=False, lambda_penalty=1.0):
        """
        Fits extreme lines to data using a soft-margin algorithm based on uncertainty rectangles.
        First, calculates the best fit using weighted least squares. Then, evaluates geometrically 
        possible extreme lines, penalizing them for missing the uncertainty bounds of the points.
        
        Args:
            x_col, y_col (int): Data column numbers (1-indexed).
            xsigma_col, ysigma_col (int): Uncertainty column numbers (1-indexed).
            naive_b (bool, optional): If True, b_avg is calculated as average of b_max and b_min. 
                                      Defaults to False (uses best fit b).
            lambda_penalty (float, optional): Weight of the penalty for missing error bars. 
                                              Higher = stricter fit. Defaults to 1.0.
          
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

        if len(x) < 2:
            raise ValueError("At least two data points are required.")

        # KROK 1: Prosta najlepszego dopasowania (Best Fit)
        def linear_model(x_val, a_val, b_val):
            return a_val * x_val + b_val

        try:
            if np.all(dy > 0):
                popt, _ = curve_fit(linear_model, x, y, sigma=dy, absolute_sigma=True)
            else:
                popt, _ = curve_fit(linear_model, x, y)
            a_avg, b_avg = popt
        except Exception as e:
            raise RuntimeError(f"Error calculating best fit: {e}")

        # KROK 2: Generowanie potencjalnych prostych skrajnych
        corners = []
        for xi, yi, dxi, dyi in zip(x, y, dx, dy):
            # 4 rogi dla każdego punktu pomiarowego
            corners.append([
                (xi + dxi, yi + dyi), (xi + dxi, yi - dyi),
                (xi - dxi, yi + dyi), (xi - dxi, yi - dyi)
            ])

        candidates = []
        # Przechodzimy przez każdą unikalną parę punktów
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                for cx1, cy1 in corners[i]:
                    for cx2, cy2 in corners[j]:
                        if cx1 == cx2:
                            continue # Pomijamy pionowe proste
                        a = (cy2 - cy1) / (cx2 - cx1)
                        b = cy1 - a * cx1
                        candidates.append((a, b))

        if not candidates:
            raise ValueError("Could not generate any valid extreme line candidates.")

        candidates = np.array(candidates)
        a_vals = candidates[:, 0]
        b_vals = candidates[:, 1]

        # KROK 3: Płynna miara odległości i Funkcja Kary (Soft-Margin)
        # Zabezpieczenie przed dzieleniem przez zero
        dy_safe = np.maximum(dy, 1e-10)
        dx_safe = np.maximum(dx, 1e-10)

        penalties = np.zeros(len(candidates))
        
        for idx, (a, b) in enumerate(candidates):
            # Odległość każdego punktu od prostej w jednostkach (efektywnej) sigmy
            effective_sigma = np.sqrt(dy_safe**2 + (a * dx_safe)**2)
            distances = np.abs(y - (a * x + b)) / effective_sigma
            
            # Kara tylko za przekroczenie marginesu (distance > 1)
            misses = np.maximum(0, distances - 1)
            penalties[idx] = np.sum(misses**2)

        # KROK 4: Znormalizowana funkcja celu
        # Normalizujemy a i kary do przedziału [0, 1], aby lambda_penalty miało stały sens
        a_ptp = np.ptp(a_vals) if np.ptp(a_vals) > 0 else 1.0
        a_norm = (a_vals - np.min(a_vals)) / a_ptp

        p_ptp = np.ptp(penalties) if np.ptp(penalties) > 0 else 1.0
        p_norm = (penalties - np.min(penalties)) / p_ptp

        # Poszukiwanie a_max (maksymalizujemy znormalizowane a, minimalizujemy karę)
        score_max = a_norm - lambda_penalty * p_norm
        best_max_idx = np.argmax(score_max)
        a_max, b_max = candidates[best_max_idx]

        # Poszukiwanie a_min (minimalizujemy znormalizowane a, minimalizujemy karę)
        # Czym mniejsze a, tym większe -a_norm
        score_min = -a_norm - lambda_penalty * p_norm
        best_min_idx = np.argmax(score_min)
        a_min, b_min = candidates[best_min_idx]

        # KROK 5: Wsteczna kompatybilność dla b
        if naive_b:
            b_avg = (b_max + b_min) / 2.0

        return [[a_max, b_max], [a_min, b_min], [a_avg, b_avg]]

class FittingDataSet(BaseDataSet, FittingMixin):
    """
    A DataSet class that includes fitting capabilities.
    """
    pass