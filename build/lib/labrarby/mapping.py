import struct
import zlib
from pathlib import Path
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy import ndimage


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

    @staticmethod
    def _read_origin_coords(tail: bytes) -> tuple[float, float, float] | None:
        """Read DataOriginValue0 (X), DataOriginValue1 (Y), and optional DataOriginValue2 (Z) from a DataList tail."""
        vals = []
        for k in range(3):
            key = f"DataOriginValue{k}".encode("utf-16le") + b"\x00\x00\x05\x00"
            idx = tail.find(key)
            if idx == -1 or idx + len(key) + 8 > len(tail):
                if k < 2:
                    return None  # Brak X lub Y
                vals.append(0.0)  # Mapa 2D (brak Z)
            else:
                vals.append(struct.unpack_from("<d", tail, idx + len(key))[0])
        return vals[0], vals[1], vals[2]

    def _init_from_wxd(self, filepath: str):
        blocks = self._extract_zlib_blocks(Path(filepath).read_bytes())

        direct_arrays = [a for b in blocks if (a := self._read_direct_vt_r8(b)) is not None]
        datalist_blocks = [b for b in blocks if b"D\x00a\x00t\x00a\x00L\x00i\x00s\x00t" in b[:64]]
        if not direct_arrays or not datalist_blocks:
            raise ValueError(f"Could not find spectral map data in '{filepath}'.")

        # Dopasuj oś spektralną (długość fali)
        spectral_axis = None
        for arr in reversed(direct_arrays):
            if self._read_embedded_vt_r8(datalist_blocks[0], len(arr)) is not None:
                spectral_axis = arr
                break
        if spectral_axis is None:
            raise ValueError("Could not match spectral axis to DataList blocks.")

        spec_len = len(spectral_axis)
        marker = struct.pack("<Q", spec_len) + b"\x05\x00"

        # Wyciągnij (X, Y, Z) i widmo bezpośrednio z każdego bloku DataList
        points = []
        for b in datalist_blocks:
            idx = b.find(marker)
            if idx == -1 or len(b) < idx + 8 + spec_len * 10:
                continue
            arr = np.frombuffer(b, dtype=VT_R8_DTYPE, count=spec_len, offset=idx + 8)
            if not np.all(arr["tag"] == 5):
                continue

            tail = b[idx + 8 + spec_len * 10 :]
            coords = self._read_origin_coords(tail)
            if coords is not None:
                points.append((coords[0], coords[1], coords[2], arr["val"].copy()))

        if not points:
            raise ValueError("This .wxd file does not contain map coordinates (DataOriginValue).")

        x_c = np.round([p[0] for p in points], 6)
        y_c = np.round([p[1] for p in points], 6)
        z_c = np.round([p[2] for p in points], 6)

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
        for i, (_, _, _, spec) in enumerate(points):
            self.volume_data[z_idx[i], :, y_idx[i], x_idx[i]] = spec

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
    
# --- AUTOMATIC EMITTER DETECTION ---

    @staticmethod
    def _gauss_linear(x: np.ndarray, amp: float, x0: float, sigma: float, slope: float, intercept: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + slope * (x - x0) + intercept

    def _fit_gauss_linear(
        self, spec_diff: np.ndarray, peak_idx: int, fwhm_nm: float, fit_half_width_factor: float = 2.5
    ) -> dict | None:
        """
        Fits a Gaussian + linear baseline around the peak on the background-subtracted spectrum.
        Returns a dict with fit parameters and plot curves (or None if fitting fails).
        """
        s = np.nan_to_num(spec_diff, nan=0.0)
        wl = self.wavelengths
        d_lambda = float(np.median(np.abs(np.diff(wl)))) if len(wl) > 1 else 1.0

        half_win_nm = max(fit_half_width_factor * max(fwhm_nm, 0.5), 7.5 * d_lambda)
        x0_init = float(wl[peak_idx])
        mask = np.abs(wl - x0_init) <= half_win_nm
        if np.sum(mask) < 6:
            return None

        x_win = wl[mask]
        y_win = s[mask]

        order = np.argsort(x_win)
        x_win = x_win[order]
        y_win = y_win[order]

        n_edge = max(2, len(y_win) // 6)
        y_left = float(np.median(y_win[:n_edge]))
        y_right = float(np.median(y_win[-n_edge:]))
        x_left = float(np.mean(x_win[:n_edge]))
        x_right = float(np.mean(x_win[-n_edge:]))

        slope_init = (y_right - y_left) / (x_right - x_left) if x_right != x_left else 0.0
        intercept_init = 0.5 * (y_left + y_right)
        amp_init = max(float(np.max(y_win)) - intercept_init, 1.0)
        sigma_init = max(fwhm_nm / 2.35482, d_lambda)

        p0 = [amp_init, x0_init, sigma_init, slope_init, intercept_init]
        bounds = (
            [0.0, float(x_win[0]), 0.5 * d_lambda, -np.inf, -np.inf],
            [np.inf, float(x_win[-1]), float(x_win[-1] - x_win[0]), np.inf, np.inf],
        )

        try:
            popt, _ = curve_fit(self._gauss_linear, x_win, y_win, p0=p0, bounds=bounds, maxfev=4000)
        except Exception:
            return None

        amp, x0, sigma, slope, intercept = [float(v) for v in popt]
        x_plot = np.linspace(x_win[0], x_win[-1], 200)
        y_total = self._gauss_linear(x_plot, amp, x0, sigma, slope, intercept)
        y_linear = slope * (x_plot - x0) + intercept

        return {
            "amp": amp,
            "x0": x0,
            "sigma": sigma,
            "fwhm_nm": 2.35482 * sigma,
            "slope": slope,
            "intercept": intercept,
            "x_plot": x_plot,
            "y_total": y_total,
            "y_linear": y_linear,
        }

    def _local_ring_background(self, z_idx: int, y_idx: int, x_idx: int, r_in: int = 2, r_out: int = 4) -> np.ndarray:
        """Computes local background spectrum as the median of an annulus around (y_idx, x_idx) at layer z_idx."""
        yy, xx = np.ogrid[: self.ny, : self.nx]
        dist2 = (yy - y_idx) ** 2 + (xx - x_idx) ** 2
        mask = (dist2 >= r_in**2) & (dist2 <= r_out**2)

        layer = self.volume_data[z_idx]  # [wave, y, x]
        if np.any(mask):
            ring_spectra = layer[:, mask]
            if not np.all(np.isnan(ring_spectra)):
                return np.nanmedian(ring_spectra, axis=1)

        return np.nanmedian(layer, axis=(1, 2))

    def _estimate_fwhm_nm(self, spec: np.ndarray, peak_idx: int) -> float:
        """Estimates peak FWHM (in nm) around peak_idx using linear interpolation."""
        s = ndimage.gaussian_filter1d(np.nan_to_num(spec, nan=0.0), sigma=1.0)
        peak_val = s[peak_idx]
        if peak_val <= 0:
            return 0.0

        half_max = 0.5 * peak_val
        n = len(s)

        left = peak_idx
        while left > 0 and s[left] > half_max:
            left -= 1
        if left < peak_idx and s[left + 1] != s[left]:
            frac = (half_max - s[left]) / (s[left + 1] - s[left])
            wl_left = self.wavelengths[left] + frac * (self.wavelengths[left + 1] - self.wavelengths[left])
        else:
            wl_left = self.wavelengths[left]

        right = peak_idx
        while right < n - 1 and s[right] > half_max:
            right += 1
        if right > peak_idx and s[right - 1] != s[right]:
            frac = (half_max - s[right]) / (s[right - 1] - s[right])
            wl_right = self.wavelengths[right] + frac * (self.wavelengths[right - 1] - self.wavelengths[right])
        else:
            wl_right = self.wavelengths[right]

        return float(abs(wl_right - wl_left))

    def _measure_peak_counts(
        self, spec_diff: np.ndarray, peak_idx: int, fwhm_nm: float
    ) -> tuple[float, float]:
        """Returns (peak_counts, integrated_counts) from the background-subtracted spectrum."""
        s = np.nan_to_num(spec_diff, nan=0.0)
        d_lambda = float(np.median(np.abs(np.diff(self.wavelengths)))) if len(self.wavelengths) > 1 else 1.0

        fwhm_px = (fwhm_nm / d_lambda) if (fwhm_nm > 0 and d_lambda > 0) else 6.0
        smooth_sigma = float(np.clip(fwhm_px / 6.0, 1.0, 3.0))
        s_smooth = ndimage.gaussian_filter1d(s, sigma=smooth_sigma)

        search_r = max(2, int(round(fwhm_px / 2.0)))
        lo_s = max(0, peak_idx - search_r)
        hi_s = min(len(s), peak_idx + search_r + 1)
        peak_counts = float(np.max(s_smooth[lo_s:hi_s]))

        half_win_px = max(3, int(round(1.5 * fwhm_px)))
        lo_i = max(0, peak_idx - half_win_px)
        hi_i = min(len(s), peak_idx + half_win_px + 1)
        integrated_counts = float(np.sum(s[lo_i:hi_i]))

        return peak_counts, integrated_counts

    def find_emitters(
        self,
        snr_threshold: float = 6.0,
        min_fwhm_nm: float | None = 1.0,
        spectral_sigma: float | None = None,
        xy_window: int = 3,
        z_window: int = 3,
        remove_cosmic_rays: bool = True,
        cosmic_sigma: float = 8.0,
        bg_ring: tuple[int, int] = (2, 4),
        fit_window_fwhm: float = 2.5,
    ) -> list[dict]:
        """
        Automatically detects emitters in a 2D or 3D spectral map with minimum FWHM filtering
        and Gaussian + linear baseline fitting around the peak.
        """
        vol = self.volume_data.copy()  # [nz, n_wave, ny, nx]
        d_lambda = float(np.median(np.abs(np.diff(self.wavelengths)))) if len(self.wavelengths) > 1 else 1.0
        min_fwhm_px = (min_fwhm_nm / d_lambda) if (min_fwhm_nm and d_lambda > 0) else 0.0

        # 1. Cosmic ray removal
        if remove_cosmic_rays:
            cr_win = 7 if min_fwhm_px == 0 or min_fwhm_px >= 8 else (5 if min_fwhm_px >= 5 else 3)
            med_spec = ndimage.median_filter(np.nan_to_num(vol, nan=0.0), size=(1, cr_win, 1, 1))
            diff = vol - med_spec
            mad_cr = 1.4826 * np.nanmedian(np.abs(diff), axis=1, keepdims=True)
            mad_cr = np.where(mad_cr < 1e-9, 1.0, mad_cr)
            cosmic_mask = diff > (cosmic_sigma * mad_cr)
            vol[cosmic_mask] = med_spec[cosmic_mask]

        # 2. Global background subtraction
        global_bg = np.nanmedian(vol, axis=(2, 3), keepdims=True)
        delta_i = vol - global_bg

        # 3. Noise normalization (SNR via MAD)
        med_delta = np.nanmedian(delta_i, axis=(2, 3), keepdims=True)
        noise = 1.4826 * np.nanmedian(np.abs(delta_i - med_delta), axis=(2, 3), keepdims=True)
        noise = np.where((noise < 1e-9) | np.isnan(noise), 1.0, noise)
        snr_4d = np.nan_to_num(delta_i / noise, nan=0.0)

        # 4. Spectral smoothing
        if spectral_sigma is None:
            sigma = float(np.clip(min_fwhm_px / 4.0, 1.0, 6.0)) if min_fwhm_px > 0 else 1.0
        else:
            sigma = spectral_sigma

        if sigma > 0:
            snr_4d = ndimage.gaussian_filter1d(snr_4d, sigma=sigma, axis=1)

        # 5. 4D -> 3D reduction
        peak_wave_idx = np.argmax(snr_4d, axis=1)  # [nz, ny, nx]
        snr_3d = np.max(snr_4d, axis=1)  # [nz, ny, nx]

        # 6. 3D local maxima
        footprint = (min(z_window, self.nz), xy_window, xy_window)
        local_max = ndimage.maximum_filter(snr_3d, size=footprint, mode="nearest")
        candidates = (snr_3d == local_max) & (snr_3d >= snr_threshold)

        coords = np.argwhere(candidates)  # [z, y, x]
        emitters = []

        for z_i, y_i, x_i in coords:
            w_i = int(peak_wave_idx[z_i, y_i, x_i])
            snr_val = float(snr_3d[z_i, y_i, x_i])

            spec_emitter = self.volume_data[z_i, :, y_i, x_i].copy()
            spec_bg = self._local_ring_background(z_i, y_i, x_i, r_in=bg_ring[0], r_out=bg_ring[1])
            spec_diff = spec_emitter - spec_bg

            # 7. FWHM verification, peak counts, and Gaussian + linear fit
            fwhm_nm = self._estimate_fwhm_nm(delta_i[z_i, :, y_i, x_i], w_i)
            if min_fwhm_nm is not None and fwhm_nm < min_fwhm_nm:
                continue

            peak_counts, integrated_counts = self._measure_peak_counts(spec_diff, w_i, fwhm_nm)
            fit = self._fit_gauss_linear(spec_diff, w_i, fwhm_nm, fit_half_width_factor=fit_window_fwhm)
            gauss_height = fit["amp"] if fit is not None else np.nan

            emitters.append(
                {
                    "z_idx": int(z_i),
                    "y_idx": int(y_i),
                    "x_idx": int(x_i),
                    "x_um": float(self.x_values[x_i]) if self.x_values is not None else float(x_i),
                    "y_um": float(self.y_values[y_i]) if self.y_values is not None else float(y_i),
                    "z_um": float(self.z_values[z_i]),
                    "wave_idx": w_i,
                    "wavelength_nm": float(self.wavelengths[w_i]),
                    "fwhm_nm": fwhm_nm,
                    "peak_counts": peak_counts,
                    "integrated_counts": integrated_counts,
                    "gauss_height": gauss_height,
                    "fit": fit,
                    "fit_window_fwhm": fit_window_fwhm,
                    "snr": snr_val,
                    "spectrum": spec_emitter,
                    "background": spec_bg,
                    "difference": spec_diff,
                }
            )

        emitters.sort(key=lambda e: e["snr"], reverse=True)
        for idx, em in enumerate(emitters):
            em["id"] = idx

        self.emitters = emitters

        print(f"Found {len(emitters)} emitters (SNR >= {snr_threshold}, FWHM >= {min_fwhm_nm} nm):")
        for em in emitters[:15]:
            g_str = f"{em['gauss_height']:6.0f}" if not np.isnan(em["gauss_height"]) else "   N/A"
            print(
                f"  [#{em['id']:02d}] SNR={em['snr']:5.1f} | "
                f"λ={em['wavelength_nm']:6.2f} nm | FWHM={em['fwhm_nm']:4.2f} nm | "
                f"Gauss={g_str} cts | Peak={em['peak_counts']:6.0f} cts | "
                f"X={em['x_idx']:2d} ({em['x_um']:.1f} µm), "
                f"Y={em['y_idx']:2d} ({em['y_um']:.1f} µm), "
                f"Z={em['z_idx']} ({em['z_um']:.2f} µm)"
            )
        if len(emitters) > 15:
            print(f"  ... and {len(emitters) - 15} more.")

        return emitters

    def show_emitters(self, **find_kwargs):
        """
        Opens an interactive window to browse detected emitters:
        - Left: 2D intensity map at optimal Z and peak wavelength with markers.
        - Right: Emitter spectrum, local background, difference, and Gaussian + linear fit.
        """
        if not hasattr(self, "emitters") or find_kwargs:
            self.find_emitters(**find_kwargs)

        if not self.emitters:
            print("No emitters to display. Try lowering 'snr_threshold'.")
            return

        is_volumetric = self.nz > 1
        fig = plt.figure(figsize=(14.5, 7.2))

        bottom = 0.22 if is_volumetric else 0.16
        height = 0.68 if is_volumetric else 0.74

        # Wider gap between colorbar (ends ~0.40) and spectrum y-axis (starts at 0.53)
        ax_map = fig.add_axes([0.05, bottom, 0.32, height])
        ax_cb = fig.add_axes([0.385, bottom, 0.016, height])
        ax_spec = fig.add_axes([0.53, bottom, 0.43, height])

        ax_prev = fig.add_axes([0.05, 0.04, 0.14, 0.055])
        ax_next = fig.add_axes([0.21, 0.04, 0.14, 0.055])
        self._btn_prev = Button(ax_prev, "← Previous")
        self._btn_next = Button(ax_next, "Next →")

        if is_volumetric:
            ax_slider_z = fig.add_axes([0.53, 0.05, 0.43, 0.035])
            self._em_slider_z = Slider(ax_slider_z, "Z Layer", 0, self.nz - 1, valinit=0, valstep=1)
        else:
            self._em_slider_z = None

        state = {"idx": 0}
        em0 = self.emitters[0]

        im = ax_map.imshow(self.volume_data[em0["z_idx"], em0["wave_idx"]], cmap="viridis", aspect="equal")
        cb = fig.colorbar(im, cax=ax_cb)
        cb.set_label("Counts")

        all_x = [e["x_idx"] for e in self.emitters]
        all_y = [e["y_idx"] for e in self.emitters]
        ax_map.scatter(all_x, all_y, s=35, facecolors="none", edgecolors="white", alpha=0.5, lw=1.0, label="Other emitters")
        (active_marker,) = ax_map.plot([em0["x_idx"]], [em0["y_idx"]], "o", mfc="none", mec="red", ms=12, mew=2.2)

        (line_em,) = ax_spec.plot(self.wavelengths, em0["spectrum"], color="tab:blue", lw=1.3, label="Emitter")
        (line_bg,) = ax_spec.plot(self.wavelengths, em0["background"], color="gray", lw=1.1, ls="--", label="Local background")
        (line_diff,) = ax_spec.plot(self.wavelengths, em0["difference"], color="tab:red", lw=1.1, alpha=0.75, label="Difference (Emitter - Bg)")
        (line_fit,) = ax_spec.plot([], [], color="black", lw=2.0, label="Fit (Gauss + linear)")
        (line_lin,) = ax_spec.plot([], [], color="black", lw=1.3, ls="--", alpha=0.85, label="Linear baseline")
        peak_vline = ax_spec.axvline(em0["wavelength_nm"], color="tab:red", ls=":", alpha=0.5)

        ax_spec.set_xlim(self.wavelengths.min(), self.wavelengths.max())
        ax_spec.set_xlabel("Wavelength [nm]")
        ax_spec.set_ylabel("Counts")
        ax_spec.grid(True, linestyle="--", alpha=0.6)
        ax_spec.legend(loc="upper right", fontsize=9)

        def render(z_override=None):
            em = self.emitters[state["idx"]]
            z_i = em["z_idx"] if z_override is None else int(z_override)
            w_i = em["wave_idx"]

            img = self.volume_data[z_i, w_i]
            im.set_data(img)
            if not np.all(np.isnan(img)):
                im.set_clim(vmin=np.nanmin(img), vmax=np.nanmax(img))

            z_info = f" | Z={self.z_values[z_i]:.2f} µm" + (" (opt)" if z_i == em["z_idx"] else "") if is_volumetric else ""
            ax_map.set_title(
                f"Emitter #{em['id']} ({state['idx'] + 1}/{len(self.emitters)}) | λ={em['wavelength_nm']:.2f} nm{z_info}",
                fontsize=11,
            )
            active_marker.set_data([em["x_idx"]], [em["y_idx"]])

            spec_curr = self.volume_data[z_i, :, em["y_idx"], em["x_idx"]]
            bg_curr = em["background"] if z_i == em["z_idx"] else self._local_ring_background(z_i, em["y_idx"], em["x_idx"])
            diff_curr = spec_curr - bg_curr

            line_em.set_ydata(spec_curr)
            line_bg.set_ydata(bg_curr)
            line_diff.set_ydata(diff_curr)
            peak_vline.set_xdata([em["wavelength_nm"], em["wavelength_nm"]])

            fit_curr = em["fit"] if z_i == em["z_idx"] else self._fit_gauss_linear(
                diff_curr, w_i, em["fwhm_nm"], fit_half_width_factor=em["fit_window_fwhm"]
            )
            if fit_curr is not None:
                line_fit.set_data(fit_curr["x_plot"], fit_curr["y_total"])
                line_lin.set_data(fit_curr["x_plot"], fit_curr["y_linear"])
                g_txt = f"Gauss={fit_curr['amp']:.0f} cts"
            else:
                line_fit.set_data([], [])
                line_lin.set_data([], [])
                g_txt = "Gauss=N/A"

            all_vals = np.concatenate([spec_curr, bg_curr, diff_curr])
            if not np.all(np.isnan(all_vals)):
                ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                margin = (ymax - ymin) * 0.06 or 1.0
                ax_spec.set_ylim(ymin - margin, ymax + margin)

            ax_spec.set_title(
                f"SNR={em['snr']:.1f} | {g_txt} | Peak={em['peak_counts']:.0f} cts | "
                f"X={em['x_idx']} ({em['x_um']:.1f} µm), Y={em['y_idx']} ({em['y_um']:.1f} µm)",
                fontsize=10.5,
            )
            fig.canvas.draw_idle()

        def select_emitter(new_idx):
            state["idx"] = new_idx % len(self.emitters)
            opt_z = self.emitters[state["idx"]]["z_idx"]
            if self._em_slider_z is not None:
                self._em_slider_z.eventson = False
                self._em_slider_z.set_val(opt_z)
                self._em_slider_z.eventson = True
            render(z_override=opt_z)

        self._btn_prev.on_clicked(lambda _: select_emitter(state["idx"] - 1))
        self._btn_next.on_clicked(lambda _: select_emitter(state["idx"] + 1))
        if self._em_slider_z is not None:
            self._em_slider_z.on_changed(lambda val: render(z_override=int(val)))

        select_emitter(0)
        plt.show()