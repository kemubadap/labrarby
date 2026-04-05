import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.signal import find_peaks
from .math_utils import create_model_function
from .base import BaseDataSet

class PlottingMixin:
    """
    Mixin class providing methods for data visualization and signal processing (FFT).
    Expects self.data to be a numpy array.
    """

    def plot_data(self, cols, meta=None, functions=None, function_params=None, do_legend=True, multiple_x_axis=False, uncertainties_columns=None, fontsize=None, Colorblind_mode=False):
        """
        Plots the data and optionally overlays mathematical models.
        
        Args:
            cols (list or dict): Data columns to plot.
            meta (tuple, optional): (filename, title, xlabel, ylabel).
            functions (list of str, optional): Math function strings to overlay.
            function_params (list of dict, optional): Parameters for the functions.
            do_legend (bool, optional): Whether to display the legend.
            multiple_x_axis (bool, optional): Treat cols as {x_col: y_col} mapping.
            uncertainties_columns (dict, optional): Mapping for error bars.
            fontsize (int, optional): Base font size for all plot elements.
            Colorblind_mode (bool, optional): If True, uses the Okabe-Ito colorblind-friendly palette.
        """
        if not isinstance(self.data, np.ndarray):
            raise TypeError("self.data must be a numpy array.")
            
        xs, ys, x_errs, y_errs = [], [], [], []

        if multiple_x_axis:
            if not isinstance(cols, dict):
                raise TypeError("When multiple_x_axis=True, 'cols' must be a dict of {x_col: y_col}.")
                
            unc_items = list(uncertainties_columns.items()) if uncertainties_columns else []
            
            for i, (x_c, y_c) in enumerate(cols.items()):
                xs.append(self.data[:, x_c - 1])
                ys.append(self.data[:, y_c - 1])
                
                if uncertainties_columns and i < len(unc_items):
                    x_unc_c, y_unc_c = unc_items[i]
                    x_e = self.data[:, x_unc_c - 1] if x_unc_c > 0 else None
                    y_e = self.data[:, y_unc_c - 1] if y_unc_c > 0 else None
                else:
                    x_e, y_e = None, None
                
                x_errs.append(x_e)
                y_errs.append(y_e)
        else:
            if isinstance(cols, list):
                x_col, *y_cols = [c - 1 for c in cols]
                x_base = self.data[:, x_col]
                
                for y_col in y_cols:
                    xs.append(x_base)
                    ys.append(self.data[:, y_col])
                    x_errs.append(None)
                    y_errs.append(None)
                    
            elif isinstance(cols, dict):
                keys = list(cols.keys())
                x_c = keys[0]
                x_unc_c = cols[x_c]
                
                x_base = self.data[:, x_c - 1]
                x_base_err = self.data[:, x_unc_c - 1] if x_unc_c > 0 else None
                
                for y_c in keys[1:]:
                    xs.append(x_base)
                    ys.append(self.data[:, y_c - 1])
                    x_errs.append(x_base_err)
                    
                    y_unc_c = cols[y_c]
                    y_errs.append(self.data[:, y_unc_c - 1] if y_unc_c > 0 else None)
            else:
                raise TypeError("When multiple_x_axis=False, 'cols' must be a list or dict.")

        if meta:
            filename, title, xlabel, ylabel = meta
        else:
            filename, title, xlabel, ylabel = None, "Plot", "X Axis", "Y Axis"

        # Konfiguracja parametrów wykresu
        rc_params = {}
        if fontsize is not None:
            rc_params.update({
                'font.size': fontsize,
                'axes.titlesize': fontsize + 2,
                'axes.labelsize': fontsize,
                'xtick.labelsize': fontsize - 1,
                'ytick.labelsize': fontsize - 1,
                'legend.fontsize': fontsize - 1
            })

        if Colorblind_mode:
            # Paleta Okabe-Ito
            cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
            rc_params['axes.prop_cycle'] = cycler(color=cb_palette)

        # Używamy menedżera kontekstu, by nie nadpisać globalnych ustawień matplotlib
        with plt.rc_context(rc_params):
            plt.figure(figsize=(8, 6))
            
            for i in range(len(ys)):
                label = f"Series {i+1}"
                if x_errs[i] is not None or y_errs[i] is not None:
                    plt.errorbar(xs[i], ys[i], xerr=x_errs[i], yerr=y_errs[i], fmt='.', label=label, capsize=3)
                else:
                    plt.scatter(xs[i], ys[i], label=label, marker='.')

            if functions and xs:
                min_x = min(np.min(x_arr) for x_arr in xs)
                max_x = max(np.max(x_arr) for x_arr in xs)
                x_range = np.linspace(min_x, max_x, 1000)
                
                for i, func in enumerate(functions):
                    param_names = function_params[i].keys() if function_params and i < len(function_params) else []
                    param_values = list(function_params[i].values()) if function_params and i < len(function_params) else []
                    
                    model_func = create_model_function(func, param_names)
                    
                    y_model = np.array([model_func(x, *param_values) for x in x_range])
                    plt.plot(x_range, y_model, label=f"Model {i+1}", linestyle='-')

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if do_legend:
                plt.legend()
            plt.grid()
            
            if filename:
                plt.savefig(filename, format='png')
                plt.close()
            else:
                plt.show()

    def fft_peaks(self, x_col, y_col, peak_height_ratio=0.01, plot=True, fontsize=None, Colorblind_mode=False):
        """
        Computes FFT and finds frequency peaks.
        
        Args:
            x_col (int): Time column (1-indexed).
            y_col (int): Signal column (1-indexed).
            peak_height_ratio (float): Minimum peak height relative to max peak.
            plot (bool): Whether to plot the spectrum.
            fontsize (int, optional): Base font size for the plot.
            Colorblind_mode (bool, optional): If True, uses colorblind-friendly colors.
        """
        if self.data is None:
            raise ValueError("No data available.")

        czas = self.data[:, x_col - 1]
        sygnal = self.data[:, y_col - 1] - np.mean(self.data[:, y_col - 1])

        N = len(sygnal)
        dt = czas[1] - czas[0] if N > 1 else 1.0

        fft_sygnal = np.fft.fft(sygnal)
        freq = np.fft.fftfreq(N, d=dt)

        mask = freq >= 0
        amplituda = np.abs(fft_sygnal)[mask]
        czestotliwosc = freq[mask]

        if len(amplituda) == 0:
            raise ValueError("Insufficient data for FFT analysis.")

        peaks, _ = find_peaks(amplituda, height=np.max(amplituda) * peak_height_ratio)

        if plot:
            rc_params = {}
            if fontsize is not None:
                rc_params.update({
                    'font.size': fontsize,
                    'axes.titlesize': fontsize + 2,
                    'axes.labelsize': fontsize,
                    'xtick.labelsize': fontsize - 1,
                    'ytick.labelsize': fontsize - 1,
                    'legend.fontsize': fontsize - 1
                })

            if Colorblind_mode:
                cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
                rc_params['axes.prop_cycle'] = cycler(color=cb_palette)

            with plt.rc_context(rc_params):
                plt.figure(figsize=(8, 4))
                plt.plot(czestotliwosc, amplituda, label="FFT Spectrum")
                
                # Używamy drugiego koloru z cyklu (jeśli Colorblind_mode jest włączony) 
                # lub po prostu czerwonego 'x' dla standardowego trybu, aby krzyżyki się wyróżniały.
                peak_color = '#56B4E9' if Colorblind_mode else 'r'
                plt.plot(czestotliwosc[peaks], amplituda[peaks], marker="x", color=peak_color, linestyle="None", label="Peaks")
                
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Amplitude")
                plt.title("Amplitude Spectrum (FFT)")
                plt.legend()
                plt.grid()
                plt.show()

        return czestotliwosc[peaks], amplituda[peaks]

class DataPlotter(BaseDataSet, PlottingMixin):
    """
    A DataSet class that includes plotting capabilities.
    """
    pass