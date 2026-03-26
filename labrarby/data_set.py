from .base import BaseDataSet
from .data_io import DataIOMixin
from .fitting import FittingMixin
from .uncertainties import UncertaintiesMixin
from .plotting import PlottingMixin


class DataSet(BaseDataSet, DataIOMixin, FittingMixin, UncertaintiesMixin, PlottingMixin):
    """
    The complete DataSet class combining IO, Fitting, Uncertainties, and Plotting.
    """
    pass