import froll
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


def _lin_model(x, a, b):
    return a * x + b


def compute_linfit(x, y, yerr=None):
    x = np.array(x)
    y = np.array(y)

    if x.size == 0 or y.size == 0:
        # Can't fit with no data; return NaNs for everything
        return np.full(2, np.nan), np.full([2,2], np.nan), np.nan

    # linear fit using y = a*x + b
    if yerr is not None:
        yerr = np.array(yerr)
        fit, cov = curve_fit(_lin_model, x, y, p0=[1, 0], sigma=yerr)
    else:
        fit, cov = curve_fit(_lin_model, x, y, p0=[1, 0])

    # pearson correlation coefficient
    R = pearsonr(x, y)[0]

    return fit, cov, R


def add_linfit(ax, x, y, yerr=None, label='{fit}', **style):
    """
    Add a line with the results of a linear fit to y = a*x + b
    Also shows the squared pearson correlation coefficient

    Inputs:
        - ax: matplotlib subplot
        - x: x axis data
        - y: y axis data
    """
    if np.size(x) == 0:
        # if no data, the min/max ops will fail - and there's nothing to
        # do anyway, so return early.
        return

    fit, cov, R = compute_linfit(x, y, yerr)

    fitstr = 'y=({:.4f} $\pm$ {:.4f})*x + ({:.4f} $\pm$ {:.4f}); R²={:.3f}'.format(
        fit[0], np.sqrt(cov[0][0]), fit[1], np.sqrt(cov[1][1]), R ** 2
    )
    label = label.format(fit=fitstr)

    # plot line fits. 
    if np.size(x) > 0:
        xfit = np.array([np.min(x), np.max(x)])
        ax.plot(xfit, _lin_model(xfit, fit[0], fit[1]), label=label, **style)


def fortran_rolling_derivatives(x, y, window, min_periods=1):
    if not isinstance(x,np.ndarray) or np.ndim(x) != 1:
        raise TypeError('x must be a 1D numpy array)')

    if not isinstance(y,np.ndarray) or np.ndim(y) != 1:
        raise TypeError('y must be a 1D numpy array)')

    if x.size != y.size:
        raise TypeError('x and y must be the same size')

    # The fortran code expects double precision; convert only if necessary
    if x.dtype != 'float64':
        x = x.astype('float64')
    if x.dtype != 'float64':
        y = y.astype('float64')

    slopes = np.full_like(x, np.nan)
    froll.rolling_linfit(x, y, window, min_periods, x.size, slopes)
    return slopes



def freq_op_str(freq, op):
    if freq in _FREQ_FULL_NAMES:
        freq_name = _FREQ_FULL_NAMES[freq]
        return f'{freq_name} {op}s'
    else:
        return f'{op}s, frequency={freq}'


def ordinal_str(i):
    if 10 < i % 100 < 20:
        return f'{i}th'
    elif i % 10 == 1:
        return f'{i}st'
    elif i % 10 == 2:
        return f'{i}nd'
    elif i % 10 == 3:
        return f'{i}rd'
    else:
        return f'{i}th'
