TCCON QC Plots installation
===========================

The QC plots package requires:

* Python 3.7 or later 
* A Fortran compiler compatible with `numpy's f2py module <https://numpy.org/doc/stable/f2py/usage.html>`_. ``gfortran`` is recommended.

Additionally, the default installation scripts expect either the ``conda`` or ``micromamba`` package managers to be installed.
If they are not available, see method #4, below.


Method 1: install scripts
-------------------------

We provide ``install-conda.sh`` and ``install-micromamba.sh`` scripts. Choose the one for the package manager you prefer and have
on your system. Simplying run the script as ``./install-conda.sh`` or ``./install-micromamba.sh``. This will:

- create a conda/micromamba environment named "tccon-qc",
- install the necessary packages,
- install ``qc_plots`` itself in develop mode, and
- link the ``qc_plots`` script to the repo as ``run-qc-plots``.

After this, you can use the ``run-qc-plots`` script to call the package without needing the "tccon-qc" environment to be activated.

.. note::
    Installing in "develop" mode means that the source code for ``qc_plots`` remains in this directory, and any time you import ``qc_plots``
    from a Python instance running in the "tccon-qc" environment, it will execute this code. Thus, changes made to this directory take effect
    the next time ``qc_plots`` is imported.


Method 2: manual conda install
------------------------------

If you wish to control which conda environment QC Plots is installed into, first create the desired environment with either ``conda create -n NAME``
or ``conda create -p PATH``, then update that environment using the :file:`environment.yml` file in the QC Plots repo and the 
`instructions in the conda docs <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment>`_.
Finally, activate that environment and run ``python setup.py develop``. 

As an example, you can mimic what ``install.sh`` does with:

    conda create -n tccon-qc
    conda env update -n tccon-qc --file environment.yml
    conda activate tccon-qc
    python setup.py develop
    ln -s $(which qc_plots) run-qc-plots

Alternatively, to create an environment in the directory :file:`.env` at the current path::
    
    conda create -p ./.env
    conda env update -p ./.env --file environment.yml
    conda activate ./.env
    python setup.py develop
    ln -s $(which qc_plots) run-qc-plots

Note that the second and fourth commands in both examples assume you are running in the top directory of the QC Plots repo.

Method 3: manual micromamba install
-----------------------------------

This is similar to method 2, but with micromamba instead. Assuming you want to use the ``tccon-qc`` named environment:

    micromamba create -n tccon-qc --file environment.yml
    micromamba activate tccon-qc
    python setup.py develop
    ln -s $(which qc_plots) run-qc-plots

As in the conda approach, the first and third commands assume you are running in the top directory of this repo.


Method 4: install with pip
--------------------------

While we recommend installing the necessary dependencies with conda because it tends to work better across all platforms, you should be able to 
install everythin with pip. We *highly* recommend installing QC Plots into its own virtual environment to avoid conflicts with other packages.
To do so, and assuming you are using bash as your shell, run::

    python -m venv ./.qcenv
    source .qcenv/bin/activate
    python setup.py develop
    ln -s $(which qc_plots) run-qc-plots

This will create a virtual environment in :file:`.qcenv` in the current directory, activate that environment, and then install everything 
at once with the ``setup.py`` call. You can replace :file:`./.qcenv` in the first command with any path you wish, as long as you use that 
path in the second command.
