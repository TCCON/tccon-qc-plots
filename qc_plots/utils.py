import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from .constants import _FREQ_FULL_NAMES


def cm2inch(*tupl):
    """
    Converts reasonable units (cm) into terrible units (inches).
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def colorbar_gridspec_kws(n_colorbars):
    ratios = [100 - 4*n_colorbars] + n_colorbars * [4]
    return {'width_ratios': ratios, 'wspace': 0.4}


def preformat_string(s, **frozen_kwargs):
    """Return a string whose `format` method has certain keyword values pre-supplied.
    """

    class PreformattedString(str):
        def format(self, *args, **kwargs):
            all_kws = frozen_kwargs.copy()
            all_kws.update(kwargs)
            return super().format(*args, *args, **all_kws)

    return PreformattedString(s)


def add_linfit(ax, x, y, yerr=None, label='{fit}', **style):
    """
    Add a line with the results of a linear fit to y = a*x + b
    Also shows the squared pearson correlation coefficient

    Inputs:
        - ax: matplotlib subplot
        - x: x axis data
        - y: y axis data
    """

    def lin_model(x, a, b):
        return a * x + b

    x = np.array(x)
    y = np.array(y)

    # linear fit using y = a*x + b
    if yerr:
        yerr = np.array(yerr)
        fit, cov = curve_fit(lin_model, x, y, p0=[1, 0], sigma=yerr)
    else:
        fit, cov = curve_fit(lin_model, x, y, p0=[1, 0])

    # pearson correlation coefficient
    R = pearsonr(x, y)[0]

    fitstr = 'y=({:.4f} $\pm$ {:.4f})*x + ({:.4f} $\pm$ {:.4f}); R²={:.3f}'.format(
        fit[0], np.sqrt(cov[0][0]), fit[1], np.sqrt(cov[1][1]), R ** 2
    )
    label = label.format(fit=fitstr)

    # plot line fits
    ax.plot(x, lin_model(x, fit[0], fit[1]), label=label, **style)


def freq_op_str(freq, op):
    if freq in _FREQ_FULL_NAMES:
        freq_name = _FREQ_FULL_NAMES[freq]
        return f'{freq_name} {op}s'
    else:
        return f'{op}s, frequency={freq}'