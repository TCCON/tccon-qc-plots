from setuptools import setup, find_packages

setup(
        name='tccon_qc_plots',
        description='Create static QA/QC plots for TCCON data',
        author='Sebastien Roche & Joshua Laughner',
        author_email='sebastien.roche@mail.utoronto.ca; jlaugh@caltech.edu',
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
        entry_points={'console_scripts': [
                'qc_plots=qc_plots.__main__:main'
            ]},
        python_requires='>=3.7'
)
