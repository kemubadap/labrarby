import numpy as np
import sympy as sp

def create_model_function(function_expr, param_names):
    """
    Creates a callable model function from a string expression.
    """
    def model(x, *params):
        local_vars = dict(zip(param_names, params))
        local_vars['x'] = x
        local_vars['np'] = np
        return eval(function_expr, {"__builtins__": {}}, local_vars)
    return model

def create_partial_derivative_function(function_expr, param_name, param_names=None):
    """
    Creates a partial derivative function for error propagation using SymPy.
    """
    if param_names is None:
        param_names = [param_name]

    symbols = {name: sp.Symbol(name) for name in param_names}
    clean_expr = function_expr.replace('np.', '')
    
    replacements = {'arctan': 'atan', 'arcsin': 'asin', 'arccos': 'acos'}
    for np_name, sp_name in replacements.items():
        clean_expr = clean_expr.replace(np_name, sp_name)

    try:
        expr = sp.sympify(clean_expr, locals=symbols)
        deriv_expr = sp.diff(expr, symbols[param_name])
        func = sp.lambdify([symbols[name] for name in param_names], deriv_expr, modules=["numpy"])
        return func
    except Exception as e:
        raise ValueError(f"Error processing derivative for '{param_name}': {e}")

def create_uncertainty_function(partial_derivatives):
    """
    Creates a combined uncertainty function using partial derivatives.
    """
    def unc_func(values, sigmas):
        if len(values) != len(sigmas) or len(values) != len(partial_derivatives):
            raise ValueError("Mismatch in lengths of values, uncertainties, and derivatives.")
        
        total = sum((deriv_func(*values) * sigma) ** 2 
                    for deriv_func, values, sigma in zip(partial_derivatives, [values]*len(values), sigmas))
        return np.sqrt(total)
    return unc_func