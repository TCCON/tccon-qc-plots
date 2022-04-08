import setuptools  # this import is required though not used directly to give the numpy setup additional features
from numpy.distutils.core import Extension

froll_ext = Extension(name='froll', sources=['fortran/froll.f'])


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(
            name='tccon_qc_plots',
            description='Create static QA/QC plots for TCCON data',
            author='Sebastien Roche & Joshua Laughner',
            author_email='sebastien.roche@mail.utoronto.ca; josh.laughner@jpl.nasa.gov',
            version='2.0.0',
            url='https://bitbucket.org/rocheseb/tccon_qc_plots/',
            install_requires=[
                    'matplotlib>=3.3.2',
                    'netcdf4>=1.5.4',
                    'numpy',
                    'pandas',
                    'pillow',
                    'pypdf2',
                    'scipy>=1.6.1',
                    'tomli>=1.0.4'
                ],
            packages=['qc_plots'],
            ext_modules=[froll_ext],
            entry_points={'console_scripts': [
                    'qc_plots=qc_plots.__main__:main',
                    'xluft_report=qc_plots.report_xluft_out_of_bounds:main'
                ]},
            python_requires='>=3.7'
    )
