# setup.py
from setuptools import setup

setup(
    name='barcgp',
    version='0.1',
    packages=['barcgp'],
    install_requires=['numpy >= 1.19.5',
                      'matplotlib>=3.1.2',
                      'scipy>=1.4.1',
                      'casadi>=3.5.1',
                      'torch>=1.10.0',
                      'gpytorch>=1.5.1',
                      'pyqtgraph>=0.12.2']
)