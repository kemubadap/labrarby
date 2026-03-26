import numpy as np
from .math_utils import (
    create_model_function,
    create_partial_derivative_function,
    create_uncertainty_function
)
from .base import BaseDataSet

class UncertaintiesMixin:
    """
    Mixin class providing methods for calculating uncertainties and adding computed columns.
    Expects self.data to be a numpy array.
    """

    def add_computed_column(self, formula, column_mapping):
        """
        Adds a new column based on a mathematical formula and mapped input columns.
        
        Args:
            formula (str): Formula expression, e.g., "x*y + np.sin(x)".
            column_mapping (dict): Mapping of variables to column numbers (1-indexed), e.g., {"x": 1, "y": 2}.
        """
        if self.data is None:
            raise ValueError("No data available.")

        param_names = list(column_mapping.keys())
        
        # Call the imported function instead of self._create_model_function
        model_func = create_model_function(formula, param_names)

        try:
            inputs = []
            for name in param_names:
                col_num = column_mapping[name]
                inputs.append(self.data[:, col_num - 1] if col_num > 0 else np.zeros(self.data.shape[0]))
            inputs = [np.asarray(arr) for arr in inputs]
        except IndexError:
            raise IndexError("Invalid column index in mapping.")

        try:
            result_column = model_func(None, *inputs)
        except Exception as e:
            raise RuntimeError(f"Error computing new column: {e}")

        result_column = np.atleast_1d(result_column)
        if result_column.ndim > 1:
            result_column = result_column.flatten()
        if result_column.shape[0] != self.data.shape[0]:
            raise ValueError("Computed column length does not match dataset length.")

        self.data = np.column_stack((self.data, result_column))
        print(f"Computed column added. New shape: {self.data.shape}")

    def add_weighted_column(self, columns_dict):
        """
        Adds two columns: weighted mean and the P-value (uncertainty measure).
        
        Args:
            columns_dict (dict): Dictionary mapping data columns to their uncertainty columns (1-indexed).
        """
        if self.data is None:
            raise ValueError("No data available.")
        if len(columns_dict) < 2:
            raise ValueError("At least two columns are required to calculate weighted values.")

        try:
            data_cols = np.array([self.data[:, col - 1] for col in columns_dict.keys()])
            uncertainty_cols = np.array([self.data[:, unc - 1] for unc in columns_dict.values()])
        except IndexError:
            raise IndexError("Invalid column numbers provided.")

        if np.any(uncertainty_cols <= 0):
            raise ValueError("Uncertainties must be strictly positive (greater than zero).")

        weights = 1 / (uncertainty_cols ** 2)
        weighted_mean = np.sum(data_cols * weights, axis=0) / np.sum(weights, axis=0)

        first_term = np.sqrt(1 / np.sum(weights, axis=0))
        second_term = np.sqrt((1 / (len(columns_dict) - 1)) *
                              (np.sum(((data_cols - weighted_mean) / uncertainty_cols) ** 2, axis=0) /
                               np.sum(weights, axis=0)))

        p_values = np.maximum(first_term, second_term)

        self.data = np.column_stack((self.data, weighted_mean, p_values))
        print("Weighted mean and P-value columns added.")

    def add_uncertainty_column(self, function_expr, param_info, naive=False):
        """
        Adds an uncertainty column based on error propagation.
        
        Args:
            function_expr (str): Function expression, e.g., "x*y + x**2".
            param_info (dict): Mapping parameter names to (value_col, uncertainty_col). 1-indexed.
            naive (bool, optional): If True, uses numeric evaluation instead of SymPy analytical derivatives.
        """
        if self.data is None:
            raise ValueError("No data available.")

        param_names = list(param_info.keys())
        num_rows = self.data.shape[0]
        result_column = np.zeros(num_rows)
        
        if not naive:
            partial_derivatives = [create_partial_derivative_function(function_expr, p, param_names) for p in param_names]
            unc_func = create_uncertainty_function(partial_derivatives)
        else:
            base_func = create_model_function(function_expr, param_names)

        for i in range(num_rows):
            values, sigmas = [], []
            nan_found = False
            
            for param in param_names:
                val_col, sigma_col = param_info[param]
                val = 0.0 if val_col == 0 else self.data[i, val_col - 1]
                sigma = 0.0 if sigma_col == 0 else self.data[i, sigma_col - 1]
                
                if np.isnan(val) or np.isnan(sigma): nan_found = True
                values.append(val)
                sigmas.append(sigma)

            if nan_found:
                result_column[i] = np.nan
                continue

            try:
                if not naive:
                    result_column[i] = unc_func(values, sigmas)
                else:
                    total_variance = 0.0
                    for j in range(len(param_names)):
                        v_plus, v_minus = list(values), list(values)
                        v_plus[j] += sigmas[j]
                        v_minus[j] -= sigmas[j]
                        
                        f_plus = base_func(None, *v_plus)
                        f_minus = base_func(None, *v_minus)
                        total_variance += (0.5 * abs(f_plus - f_minus)) ** 2
                        
                    result_column[i] = np.sqrt(total_variance)
            except Exception as e:
                print(f"Warning: Error computing uncertainty at row {i}: {e}")
                result_column[i] = np.nan

        self.data = np.column_stack((self.data, result_column))
        method = "Naive" if naive else "Analytical (SymPy)"
        print(f"Added uncertainty column using {method} method.")

class DataUncertainties(BaseDataSet, UncertaintiesMixin):
    """
    A DataSet class that includes uncertainty calculation capabilities.
    """
    pass