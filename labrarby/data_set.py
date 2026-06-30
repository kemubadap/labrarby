from .base import BaseDataSet
from .data_io import DataIOMixin
from .fitting import FittingMixin
from .uncertainties import UncertaintiesMixin
from .plotting import PlottingMixin


class DataSet(BaseDataSet, DataIOMixin, FittingMixin, UncertaintiesMixin, PlottingMixin):
    """
    The complete DataSet class combining IO, Fitting, Uncertainties, and Plotting.
    """
    def __add__(self, other):
        if isinstance(other, DataSet) and self.data.shape[1] == other.data.shape[1]:
            combined_data = np.hstack((self.data, other.data))
            return DataSet(combined_data)
        else:
            raise TypeError("Can only add another DataSet with the same number of columns.")