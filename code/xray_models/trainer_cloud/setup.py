from setuptools import find_packages
from setuptools import setup

setup(
    name='chest_xray',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Using vision to detect diseases present in chest x-rays'
)
