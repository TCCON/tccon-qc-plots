TCCON QC Plots installation
===========================

The QC plots package requires:

* Python 3.7 or later 
* A Fortran compiler compatible with `numpy's f2py module <https://numpy.org/doc/stable/f2py/usage.html>`_. ``gfortran`` is recommended.

Additionally, installation methods 1 and 2 rely on the ``conda`` package manager found in the Anaconda or Miniconda Python distributions,
though ``conda`` is not a hard dependency. If you need to install QC Plots without ``conda``, see method #3, below.

Method 1: install.sh
--------------------

Run the ``install.sh`` script in the top directory of the QC Plots package as any other Bash script. This will create a conda environment
named "tccon-qc", install the necessary packages, and finally install ``qc_plots`` itself in develop mode. 

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

As an example, to create an environment in the directory :file:`.env` at the current path::
    
    conda create -p ./.env
    conda update -p ./.env -f environment.yml
    conda activate ./.env
    python setup.py develop

Note that the second and fourth commands assume you are running in the top directory of the QC Plots repo.

Method 3: install with pip
--------------------------

While we recommend installing the necessary dependencies with conda because it tends to work better across all platforms, you should be able to 
install everythin with pip. We *highly* recommend installing QC Plots into its own virtual environment to avoid conflicts with other packages.
To do so, and assuming you are using bash as your shell, run::

    python -m venv ./.qcenv
    source .qcenv/bin/activate
    python setup.py develop

This will create a virtual environment in :file:`.qcenv` in the current directory, activate that environment, and then install everything 
at once with the ``setup.py`` call. You can replace :file:`./.qcenv` in the first command with any path you wish, as long as you use that 
path in the second command.