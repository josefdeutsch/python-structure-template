from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='LSTM',
    version='0.1dev0',
    author='Josef', 
    author_email='josephdeutsch3d@gmail.com',
    packages=find_packages(),
    long_description=open('README.md').read()
)