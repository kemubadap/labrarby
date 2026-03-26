import numpy as np

class BaseDataSet:
    """Core class containing only the data initialization."""
    def __init__(self, filename_or_data):
        if isinstance(filename_or_data, str):
            self.filename = filename_or_data
            try:
                self.data = np.loadtxt(filename_or_data)
            except Exception as e:
                raise FileNotFoundError(f"Error loading file '{filename_or_data}': {e}")
        elif isinstance(filename_or_data, np.ndarray):
            self.filename = None
            self.data = filename_or_data
        else:
            raise TypeError("Unsupported data type. Expected a string (filename) or numpy array.")