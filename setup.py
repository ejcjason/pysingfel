from setuptools import setup
from io import open

requirements = [
    'numpy',
    'numba',
    'scipy',
    'mpi4py',
    'h5py'
]

setup(name='pysingfel',
      maintainer='Juncheng E',
      version='0.3.0',
      maintainer_email='juncheng.e@xfel.eu',
      description='SimEx version of pysingfel.',
      long_description=open('README.rst', encoding='utf8').read(),
      url='https://github.com/JunCEEE/pysingfel/tree/simex',
      packages=['pysingfel'],
      scripts=['bin/radiationDamageMPI'],
      install_requires=requirements,
      zip_safe=False)
