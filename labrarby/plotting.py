import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from cycler import cycler
from scipy.signal import find_peaks
from .math_utils import create_model_function
from .base import BaseDataSet

class PlottingMixin:
    """
    Mixin class providing methods for data visualization and signal processing (FFT).
    Expects self.data to be a numpy array.
    """

    def plot_data(self, cols, meta=None, functions=None, function_params=None, do_legend=True, multiple_x_axis=False, uncertainties_columns=None, fontsize=None, Colorblind_mode=False, plot_style='scatter', custom_legend=None):
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
            plot_style (str, optional): Style of the plot. Options: 'scatter' (default), 'line', 'both'.
            custom_legend (list of str, optional): Custom names for the plotted data series.
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

        if do_legend and custom_legend is not None:
            if not isinstance(custom_legend, list):
                raise TypeError("Parameter custom_legend must be a list of strings.")
            if len(custom_legend) != len(ys):
                raise ValueError(f"ValueError: Provided {len(custom_legend)} legend labels, but there are {len(ys)} data series to plot.")

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
            plt.figure(figsize=(8, 6))
            
            for i in range(len(ys)):
                label = custom_legend[i] if custom_legend is not None else f"Series {i+1}"
                
                if plot_style == 'line':
                    fmt = '-'
                    marker = None
                    linestyle = '-'
                elif plot_style == 'both':
                    fmt = '.-'
                    marker = '.'
                    linestyle = '-'
                else: 
                    fmt = '.'
                    marker = '.'
                    linestyle = 'None'

                if x_errs[i] is not None or y_errs[i] is not None:
                    plt.errorbar(xs[i], ys[i], xerr=x_errs[i], yerr=y_errs[i], fmt=fmt, label=label, capsize=3)
                else:
                    plt.plot(xs[i], ys[i], label=label, marker=marker, linestyle=linestyle)

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
    def __add__(self, other):
        if isinstance(other, DataPlotter) and self.data.shape[0] == other.data.shape[0]:
            combined_data = np.hstack((self.data, other.data))
            return DataPlotter(combined_data)
        else:
            raise TypeError("Can only add another DataPlotter with the same number of rows.")


class SpectralMapPlotter:
    """
    A standalone class for interactive visualization of Z-path spectral maps.
    """
    def __init__(self, filename_or_data, nx, ny, skiprows=1):
        """
        Args:
            filename_or_data (str or np.ndarray): Source data. First column must be wavelengths.
            nx (int): Number of horizontal pixels in the map.
            ny (int): Number of vertical pixels in the map.
            skiprows (int): Number of header rows to skip if reading from a file.
        """
        self.nx = nx
        self.ny = ny
        
        if isinstance(filename_or_data, str):
            self.filename = filename_or_data
            try:
                self.data = np.loadtxt(filename_or_data, skiprows=skiprows)
            except Exception as e:
                raise FileNotFoundError(f"Error loading file '{filename_or_data}': {e}")
        elif isinstance(filename_or_data, np.ndarray):
            self.filename = None
            self.data = filename_or_data
        else:
            raise TypeError("Expected filename (str) or numpy array.")
            
        self.wavelengths = self.data[:, 0]
        self.spectra = self.data[:, 1:]
        
        # Weryfikacja poprawności wymiarów mapy
        expected_pixels = self.nx * self.ny
        actual_pixels = self.spectra.shape[1]
        
        if expected_pixels != actual_pixels:
            raise ValueError(f"Shape mismatch: Zadeklarowano {expected_pixels} pikseli (NX={self.nx} * NY={self.ny}), ale dane zawierają kolumny dla {actual_pixels} pikseli.")
            
        # Reshape do postaci [fale, y, x]
        self.map_data = self.spectra.reshape(len(self.wavelengths), self.ny, self.nx)

    def show_interactive_map(self):
        """
        Launches the interactive Matplotlib window to browse the spectral map.
        """
        fig = plt.figure(figsize=(13, 7))

        # Definicje obszarów (grid)
        ax_map = fig.add_axes([0.08, 0.20, 0.55, 0.70])      
        ax_cb = fig.add_axes([0.64, 0.20, 0.02, 0.70])       
        ax_spec = fig.add_axes([0.08, 0.20, 0.55, 0.70])     
        ax_indicator = fig.add_axes([0.75, 0.20, 0.2, 0.35]) 
        ax_radio = fig.add_axes([0.75, 0.70, 0.2, 0.15])     

        ax_slider_wave = fig.add_axes([0.08, 0.08, 0.55, 0.03])
        ax_slider_pixel = fig.add_axes([0.08, 0.08, 0.55, 0.03])

        # --- KONFIGURACJA TRYBU 1: MAPA ---
        initial_wave_idx = 0
        im_map = ax_map.imshow(self.map_data[initial_wave_idx], cmap='viridis', aspect='equal')
        cb_map = fig.colorbar(im_map, cax=ax_cb)
        cb_map.set_label('Liczba zliczeń', fontsize=11)
        ax_map.set_title(f"Mapa intensywności. Długość fali: {self.wavelengths[initial_wave_idx]:.2f} nm", fontsize=14)

        # Zapisujemy widgety jako atrybuty instancji!
        self._slider_wave = Slider(ax_slider_wave, 'Krok Fali', 0, len(self.wavelengths)-1, valinit=initial_wave_idx, valstep=1)

        # --- KONFIGURACJA TRYBU 2: WIDMO I MINIMAPA ---
        initial_pixel = 0
        y_init, x_init = divmod(initial_pixel, self.nx)

        line_spec, = ax_spec.plot(self.wavelengths, self.spectra[:, initial_pixel], color='blue', lw=1.5)
        ax_spec.set_xlim(self.wavelengths.min(), self.wavelengths.max())
        ax_spec.set_xlabel('Długość fali [nm]', fontsize=11)
        ax_spec.set_ylabel('Liczba zliczeń', fontsize=11)
        ax_spec.grid(True, linestyle='--', alpha=0.7)
        ax_spec.set_title(f"Widmo dla piksela nr {initial_pixel} (Współrzędne mapy: X={x_init}, Y={y_init})", fontsize=14)

        ax_indicator.set_xlim(-0.5, self.nx - 0.5)
        ax_indicator.set_ylim(self.ny - 0.5, -0.5) 
        ax_indicator.set_aspect('equal')
        ax_indicator.set_title("Pozycja badanego piksela")
        ax_indicator.set_xticks([])
        ax_indicator.set_yticks([])

        rect = plt.Rectangle((-0.5, -0.5), self.nx, self.ny, fill=False, color='black', lw=1.5)
        ax_indicator.add_patch(rect)
        point_ind, = ax_indicator.plot([x_init], [y_init], 'ro', markersize=8)

        self._slider_pixel = Slider(ax_slider_pixel, 'Nr Piksela', 0, self.ny*self.nx - 1, valinit=initial_pixel, valstep=1)

        # --- PRZYCISKI I LOGIKA PRZEŁĄCZANIA ---
        self._radio = RadioButtons(ax_radio, ('Tryb Mapy (Fala)', 'Tryb Widma (Piksel)'))

        def switch_mode(label):
            if label == 'Tryb Mapy (Fala)':
                ax_map.set_visible(True)
                ax_cb.set_visible(True)
                ax_slider_wave.set_visible(True)
                self._slider_wave.active = True
                
                ax_spec.set_visible(False)
                ax_indicator.set_visible(False)
                ax_slider_pixel.set_visible(False)
                self._slider_pixel.active = False
            else:
                ax_spec.set_visible(True)
                ax_indicator.set_visible(True)
                ax_slider_pixel.set_visible(True)
                self._slider_pixel.active = True
                
                ax_map.set_visible(False)
                ax_cb.set_visible(False)
                ax_slider_wave.set_visible(False)
                self._slider_wave.active = False
                
            fig.canvas.draw_idle()

        self._radio.on_clicked(switch_mode)

        def update_wave(val):
            idx = int(self._slider_wave.val)
            im_map.set_data(self.map_data[idx])
            im_map.set_clim(vmin=self.map_data[idx].min(), vmax=self.map_data[idx].max())
            ax_map.set_title(f"Mapa intensywności. Długość fali: {self.wavelengths[idx]:.2f} nm", fontsize=14)
            fig.canvas.draw_idle()

        def update_pixel(val):
            idx = int(self._slider_pixel.val)
            y, x = divmod(idx, self.nx)
            
            current_spec = self.spectra[:, idx]
            line_spec.set_ydata(current_spec)
            
            margin = (current_spec.max() - current_spec.min()) * 0.05
            if margin == 0: margin = 1
            ax_spec.set_ylim(current_spec.min() - margin, current_spec.max() + margin)
            ax_spec.set_title(f"Widmo dla piksela nr {idx} (Współrzędne mapy: X={x}, Y={y})", fontsize=14)
            
            point_ind.set_data([x], [y])
            fig.canvas.draw_idle()

        self._slider_wave.on_changed(update_wave)
        self._slider_pixel.on_changed(update_pixel)

        # Inicjalizacja domyślnego trybu
        switch_mode('Tryb Mapy (Fala)')
        plt.show()