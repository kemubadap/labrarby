import numpy as np
from .base import BaseDataSet

class DataIOMixin:
    """
    Mixin class providing methods for data input, output, and basic matrix manipulation.
    Expects self.data to be a numpy array and self.filename to be a string or None.
    """

    def display_column(self, col):
        """
        Displays a single column.
        
        Args:
            col (int): Column number (1-indexed).
        """
        if self.data is None:
            raise ValueError("No data available.")
        try:
            col_data = self.data[:, col - 1]
        except IndexError:
            raise IndexError(f"Column {col} is out of bounds.")
            
        print(f"Column {col}:")
        print(col_data)

    def display_all_columns(self):
        """
        Displays all data columns.
        """
        if self.data is None:
            raise ValueError("No data available.")
        print("All columns:")
        print(self.data)

    def remove_columns(self, columns_to_remove):
        """
        Removes specified columns from the dataset.
        
        Args:
            columns_to_remove (list of int): List of column numbers to remove (1-indexed).
        """
        if self.data is None:
            raise ValueError("No data available.")

        num_cols = self.data.shape[1]
        invalid_cols = [col for col in columns_to_remove if col < 1 or col > num_cols]

        if invalid_cols:
            raise ValueError(f"Invalid column numbers: {invalid_cols}")

        self.data = np.delete(self.data, [col - 1 for col in columns_to_remove], axis=1)
        print(f"Removed columns: {columns_to_remove}. New shape: {self.data.shape}")

    def remove_rows(self, rows_to_remove):
        """
        Removes specified rows from the dataset.
        
        Args:
            rows_to_remove (list of int): List of row numbers to remove (1-indexed).
        """
        if self.data is None:
            raise ValueError("No data available to modify.")

        rem_indices = [i - 1 for i in rows_to_remove]

        try:
            rem_indices = sorted(set(rem_indices), reverse=True)
            self.data = np.delete(self.data, rem_indices, axis=0)
            print(f"Removed {len(rem_indices)} rows. New shape: {self.data.shape}")
        except IndexError as e:
            raise IndexError(f"Error removing rows: {e}")

    def save_columns_to_file(self, columns, filename):
        """
        Saves selected columns to a text file.
        
        Args:
            columns (list of int): List of column numbers to save (1-indexed).
            filename (str): Name of the output file.
        """
        if self.data is None:
            raise ValueError("No data available.")

        num_cols = self.data.shape[1]
        invalid_cols = [col for col in columns if col < 1 or col > num_cols]

        if invalid_cols:
            raise ValueError(f"Invalid column numbers: {invalid_cols}")

        columns_to_save = self.data[:, [col - 1 for col in columns]]

        try:
            np.savetxt(filename, columns_to_save, fmt='%.6f', delimiter=' ', comments='')
            print(f"Columns saved to file: {filename}")
        except Exception as e:
            raise IOError(f"Error saving to file: {e}")

    def filtered_by_not_nan(self, columns):
        """
        Returns a new instance filtering out NaN values in specific columns.
        
        Args:
            columns (list of int): List of column numbers to check for NaNs (1-indexed).
        """
        if self.data is None:
            raise ValueError("No data available.")

        col_indices = [c - 1 for c in columns]
        mask = ~np.isnan(self.data[:, col_indices]).any(axis=1)
        
        # Safe instantiation without hardcoding the class name
        new_ds = self.__class__(self.filename if self.filename else np.empty((0,0)))
        new_ds.data = self.data[mask]
        return new_ds

class DataManager(DataIOMixin, BaseDataSet):
    """
    A class that combines DataIOMixin with BaseDataSet for complete data management.
    """
    pass