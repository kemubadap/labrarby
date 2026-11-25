import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from cycler import cycler
from scipy.signal import find_peaks
from .math_utils import create_model_function
from .base import BaseDataSet
import struct
import zlib
from pathlib import Path



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


VT_R8_DTYPE = np.dtype([("tag", "<u2"), ("val", "<f8")])


class SpectralMapPlotter:
    """
    A standalone class for interactive visualization of 2D spectral maps
    and 3D volumetric spectral maps (.txt, numpy arrays, and Renishaw .wxd).
    """

    def __init__(self, filename_or_data, nx=None, ny=None, skiprows=1, scan_path="z-path"):
        """
        Args:
            filename_or_data (str, Path, or np.ndarray): Source data.
                For .txt/ndarray, first column must be wavelengths.
                For .wxd, nx, ny, and nz are inferred automatically.
            nx (int, optional): Number of horizontal pixels (required for .txt/ndarray).
            ny (int, optional): Number of vertical pixels (required for .txt/ndarray).
            skiprows (int): Number of header rows to skip if reading from a .txt file.
            scan_path (str): Scan order for .txt/ndarray maps and pixel slider ('z-path' or 'snake').
        """
        scan_path = scan_path.lower()
        if scan_path not in ("z-path", "snake"):
            raise ValueError(f"Unsupported scan_path '{scan_path}'. Expected 'z-path' or 'snake'.")
        self.scan_path = scan_path

        self.x_values = None
        self.y_values = None
        self.z_values = np.array([0.0])

        if isinstance(filename_or_data, (str, Path)):
            self.filename = str(filename_or_data)
            if self.filename.lower().endswith(".wxd"):
                self._init_from_wxd(self.filename)
                return
            try:
                self.data = np.loadtxt(self.filename, skiprows=skiprows)
            except Exception as e:
                raise FileNotFoundError(f"Error loading file '{self.filename}': {e}") from e
        elif isinstance(filename_or_data, np.ndarray):
            self.filename = None
            self.data = filename_or_data
        else:
            raise TypeError("Expected filename (str/Path) or numpy array.")

        if nx is None or ny is None:
            raise ValueError("For .txt or numpy array inputs, 'nx' and 'ny' must be provided.")

        self.nx = nx
        self.ny = ny
        self.nz = 1
        self.wavelengths = self.data[:, 0]
        self.spectra = self.data[:, 1:]

        expected_pixels = self.nx * self.ny
        actual_pixels = self.spectra.shape[1]
        if expected_pixels != actual_pixels:
            raise ValueError(
                f"Shape mismatch: Zadeklarowano {expected_pixels} pikseli "
                f"(NX={self.nx} * NY={self.ny}), ale dane zawierają kolumny dla {actual_pixels} pikseli."
            )

        # Reshape do postaci [fale, y, x]
        self.map_data = self.spectra.reshape(len(self.wavelengths), self.ny, self.nx).copy()
        if self.scan_path == "snake":
            self.map_data[:, 1::2, :] = self.map_data[:, 1::2, ::-1]

        # Unified 4D shape: [z, wave, y, x]
        self.volume_data = self.map_data[np.newaxis, :, :, :]

    # --- WXD LOADING HELPERS ---

    @staticmethod
    def _extract_zlib_blocks(raw: bytes) -> list[bytes]:
        blocks = []
        pos = 0
        while pos < len(raw) - 2:
            idx = raw.find(b"\x78\x9c", pos)
            if idx == -1:
                break
            try:
                decomp = zlib.decompressobj()
                out = decomp.decompress(raw[idx:])
                if decomp.eof and len(out) > 16:
                    consumed = len(raw[idx:]) - len(decomp.unused_data)
                    blocks.append(out)
                    pos = idx + consumed
                    continue
            except zlib.error:
                pass
            pos = idx + 1
        return blocks

    @staticmethod
    def _read_direct_vt_r8(block: bytes) -> np.ndarray | None:
        if len(block) < 18:
            return None
        count = struct.unpack_from("<Q", block, 0)[0]
        if not (2 <= count <= 10_000_000) or len(block) < 8 + count * 10:
            return None
        arr = np.frombuffer(block, dtype=VT_R8_DTYPE, count=count, offset=8)
        if np.all(arr["tag"] == 5):
            return arr["val"].copy()
        return None

    @staticmethod
    def _read_embedded_vt_r8(block: bytes, expected_len: int) -> np.ndarray | None:
        marker = struct.pack("<Q", expected_len) + b"\x05\x00"
        idx = block.find(marker)
        if idx != -1 and len(block) >= idx + 8 + expected_len * 10:
            arr = np.frombuffer(block, dtype=VT_R8_DTYPE, count=expected_len, offset=idx + 8)
            if np.all(arr["tag"] == 5):
                return arr["val"].copy()
        return None

    def _init_from_wxd(self, filepath: str):
        blocks = self._extract_zlib_blocks(Path(filepath).read_bytes())

        direct_arrays = [a for b in blocks if (a := self._read_direct_vt_r8(b)) is not None]
        datalist_blocks = [b for b in blocks if b"D\x00a\x00t\x00a\x00L\x00i\x00s\x00t" in b[:64]]
        if not direct_arrays or not datalist_blocks:
            raise ValueError(f"Could not find spectral map data in '{filepath}'.")

        spectral_axis = None
        for arr in reversed(direct_arrays):
            if self._read_embedded_vt_r8(datalist_blocks[0], len(arr)) is not None:
                spectral_axis = arr
                break
        if spectral_axis is None:
            raise ValueError("Could not match spectral axis to DataList blocks.")

        spec_len = len(spectral_axis)
        spectra = [y for b in datalist_blocks if (y := self._read_embedded_vt_r8(b, spec_len)) is not None]

        coord_candidates = [a for a in direct_arrays if len(a) != spec_len]
        if not coord_candidates:
            raise ValueError("This .wxd file appears to be a single spectrum, not a map.")

        n_points = max(
            {len(a) for a in coord_candidates},
            key=lambda L: sum(len(a) == L for a in coord_candidates),
        )
        coords = [a for a in coord_candidates if len(a) == n_points]

        # Usuwamy duplikaty tablic współrzędnych (snapshoty Metakit)
        unique_coords = []
        for c in coords:
            if not any(np.allclose(c, u, equal_nan=True) for u in unique_coords):
                unique_coords.append(c)

        if len(unique_coords) > 3:
            unique_coords = unique_coords[-3:]

        if len(spectra) < n_points:
            raise ValueError(f"Found only {len(spectra)} spectra for {n_points} map coordinates.")
        spectra = spectra[:n_points]

        x_c = np.round(unique_coords[0], 6)
        y_c = np.round(unique_coords[1], 6) if len(unique_coords) >= 2 else np.zeros(n_points)
        z_c = np.round(unique_coords[2], 6) if len(unique_coords) == 3 else np.zeros(n_points)

        self.x_values = np.unique(x_c)
        self.y_values = np.unique(y_c)
        self.z_values = np.unique(z_c)

        self.nx = len(self.x_values)
        self.ny = len(self.y_values)
        self.nz = len(self.z_values)
        self.wavelengths = spectral_axis

        x_idx = np.searchsorted(self.x_values, x_c)
        y_idx = np.searchsorted(self.y_values, y_c)
        z_idx = np.searchsorted(self.z_values, z_c)

        self.volume_data = np.full((self.nz, spec_len, self.ny, self.nx), np.nan, dtype=np.float64)
        for i in range(n_points):
            self.volume_data[z_idx[i], :, y_idx[i], x_idx[i]] = spectra[i]

        self.map_data = self.volume_data[0]
        self.spectra = self.map_data.reshape(spec_len, self.ny * self.nx)
        self.data = np.column_stack((self.wavelengths, self.spectra))

    # --- PIXEL INDEX MAPPING ---

    def _pixel_to_xy(self, pixel_idx: int) -> tuple[int, int]:
        y, x = divmod(pixel_idx, self.nx)
        if self.scan_path == "snake" and y % 2 == 1:
            x = self.nx - 1 - x
        return x, y

    # --- INTERACTIVE PLOTTER ---

    def show_interactive_map(self):
        """
        Launches the interactive Matplotlib window to browse the 2D or 3D spectral map.
        """
        is_volumetric = self.nz > 1
        fig = plt.figure(figsize=(13, 7.5 if is_volumetric else 7))

        main_bottom = 0.24 if is_volumetric else 0.20
        main_height = 0.66 if is_volumetric else 0.70

        ax_map = fig.add_axes([0.08, main_bottom, 0.55, main_height])
        ax_cb = fig.add_axes([0.64, main_bottom, 0.02, main_height])
        ax_spec = fig.add_axes([0.08, main_bottom, 0.55, main_height])
        ax_indicator = fig.add_axes([0.75, main_bottom, 0.20, 0.35])
        ax_radio = fig.add_axes([0.75, 0.70, 0.20, 0.15])

        slider_y = 0.12 if is_volumetric else 0.08
        ax_slider_wave = fig.add_axes([0.08, slider_y, 0.55, 0.03])
        ax_slider_pixel = fig.add_axes([0.08, slider_y, 0.55, 0.03])

        initial_z = 0
        initial_wave_idx = 0
        initial_pixel = 0
        x_init, y_init = self._pixel_to_xy(initial_pixel)

        def z_label(z_i):
            return f" | Z={self.z_values[z_i]:.2f} µm (warstwa {z_i})" if is_volumetric else ""

        # --- TRYB 1: MAPA ---
        slice_init = self.volume_data[initial_z, initial_wave_idx]
        im_map = ax_map.imshow(slice_init, cmap="viridis", aspect="equal")
        cb_map = fig.colorbar(im_map, cax=ax_cb)
        cb_map.set_label("Liczba zliczeń", fontsize=11)
        ax_map.set_title(
            f"Mapa intensywności. Długość fali: {self.wavelengths[initial_wave_idx]:.2f} nm{z_label(initial_z)}",
            fontsize=13,
        )

        self._slider_wave = Slider(
            ax_slider_wave, "Krok Fali", 0, len(self.wavelengths) - 1, valinit=initial_wave_idx, valstep=1
        )

        # --- TRYB 2: WIDMO I MINIMAPA ---
        spec_init = self.volume_data[initial_z, :, y_init, x_init]
        (line_spec,) = ax_spec.plot(self.wavelengths, spec_init, color="blue", lw=1.5)
        ax_spec.set_xlim(self.wavelengths.min(), self.wavelengths.max())
        ax_spec.set_xlabel("Długość fali [nm]", fontsize=11)
        ax_spec.set_ylabel("Liczba zliczeń", fontsize=11)
        ax_spec.grid(True, linestyle="--", alpha=0.7)
        ax_spec.set_title(
            f"Widmo dla piksela nr {initial_pixel} (X={x_init}, Y={y_init}){z_label(initial_z)}",
            fontsize=13,
        )

        ax_indicator.set_xlim(-0.5, self.nx - 0.5)
        ax_indicator.set_ylim(self.ny - 0.5, -0.5)
        ax_indicator.set_aspect("equal")
        ax_indicator.set_title("Pozycja badanego piksela")
        ax_indicator.set_xticks([])
        ax_indicator.set_yticks([])

        rect = plt.Rectangle((-0.5, -0.5), self.nx, self.ny, fill=False, color="black", lw=1.5)
        ax_indicator.add_patch(rect)
        (point_ind,) = ax_indicator.plot([x_init], [y_init], "ro", markersize=8)

        self._slider_pixel = Slider(
            ax_slider_pixel, "Nr Piksela", 0, self.ny * self.nx - 1, valinit=initial_pixel, valstep=1
        )

        # --- SUWAK Z (TYLKO DLA MAP WOLUMETRYCZNYCH, AKTYWNY W OBU TRYBACH) ---
        if is_volumetric:
            ax_slider_z = fig.add_axes([0.08, 0.05, 0.55, 0.03])
            self._slider_z = Slider(
                ax_slider_z, "Warstwa Z", 0, self.nz - 1, valinit=initial_z, valstep=1
            )
        else:
            self._slider_z = None

        # --- PRZYCISKI I LOGIKA PRZEŁĄCZANIA ---
        self._radio = RadioButtons(ax_radio, ("Tryb Mapy (Fala)", "Tryb Widma (Piksel)"))

        def get_z():
            return int(self._slider_z.val) if self._slider_z is not None else 0

        def switch_mode(label):
            if label == "Tryb Mapy (Fala)":
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

        def update_map_view(_=None):
            z_i = get_z()
            w_i = int(self._slider_wave.val)
            img = self.volume_data[z_i, w_i]
            im_map.set_data(img)
            if not np.all(np.isnan(img)):
                im_map.set_clim(vmin=np.nanmin(img), vmax=np.nanmax(img))
            ax_map.set_title(
                f"Mapa intensywności. Długość fali: {self.wavelengths[w_i]:.2f} nm{z_label(z_i)}",
                fontsize=13,
            )
            fig.canvas.draw_idle()

        def update_spec_view(_=None):
            z_i = get_z()
            p_i = int(self._slider_pixel.val)
            x, y = self._pixel_to_xy(p_i)

            current_spec = self.volume_data[z_i, :, y, x]
            line_spec.set_ydata(current_spec)

            if not np.all(np.isnan(current_spec)):
                s_min, s_max = np.nanmin(current_spec), np.nanmax(current_spec)
                margin = (s_max - s_min) * 0.05 or 1.0
                ax_spec.set_ylim(s_min - margin, s_max + margin)

            ax_spec.set_title(
                f"Widmo dla piksela nr {p_i} (X={x}, Y={y}){z_label(z_i)}",
                fontsize=13,
            )
            point_ind.set_data([x], [y])
            fig.canvas.draw_idle()

        def update_z(_=None):
            update_map_view()
            update_spec_view()

        self._radio.on_clicked(switch_mode)
        self._slider_wave.on_changed(update_map_view)
        self._slider_pixel.on_changed(update_spec_view)
        if self._slider_z is not None:
            self._slider_z.on_changed(update_z)

        switch_mode("Tryb Mapy (Fala)")
        plt.show()