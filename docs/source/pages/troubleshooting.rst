Troubleshooting
===============

Qt error
--------

If you get an error that refers to not being able to initialize a Qt backend (or something along those lines),
try updating matplotlib.
To do so, activate the ``tccon-qc`` environment with ``conda activate tccon-qc`` or ``micromamba activate tccon-qc``,
depending on which install script you used, then run ``conda update matplotlib`` or ``micromamba update matplotlib``.

Version conflict error
----------------------

If you get an error that mentions a version conflict of some sort regarding numpy and scipy, try updating numpy and scipy.
As in the Qt error above, activate the ``tccon-qc`` environment with the appropriate command, and then do either
``conda update numpy scipy`` or ``micromamba update numpy scipy``, depending as above on which of ``conda`` or ``micromamba``
you used to install ``qc_plots``.
