import matplotlib.ticker as ticker
from matplotlib import axes

def smart_comma_format(x, _):
    """
    Formatter that replaces decimal points with commas for axis labels.
    Removes trailing zeros for cleaner display.
    """
    if x == int(x):
        return f"{int(x)}".replace('.', ',')
    else:
        return f"{x:.2f}".rstrip('0').rstrip('.').replace('.', ',')

comma_formatter = ticker.FuncFormatter(smart_comma_format)

# Store the original constructor to prevent infinite loops if called multiple times
_original_axes_init = None

def enable_global_style():
    """
    Globally overrides matplotlib's Axes constructor to apply Labrarby's 
    custom styling (comma formatters and inward ticks) to all future plots.
    
    This replaces the default matplotlib behavior for the entire session.
    """
    global _original_axes_init
    
    # Prevent patching multiple times if the user calls the function twice
    if _original_axes_init is not None:
        return
        
    _original_axes_init = axes.Axes.__init__

    def custom_axes_init(self, *args, **kwargs):
        # Call the original constructor first
        _original_axes_init(self, *args, **kwargs)
        
        # Apply comma formatting to both axes
        self.xaxis.set_major_formatter(comma_formatter)
        self.yaxis.set_major_formatter(comma_formatter)

        # X-axis ticks: bottom only, pointing inward
        self.tick_params(
            axis='x', bottom=True, top=False, labelbottom=True, direction='in'
        )

        # Y-axis ticks: left only, pointing inward
        self.tick_params(
            axis='y', left=True, right=False, labelleft=True, direction='in'
        )

    # Apply the global patch
    axes.Axes.__init__ = custom_axes_init