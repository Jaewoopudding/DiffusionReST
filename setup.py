from setuptools import setup, find_packages

setup(
    name="ddpo_pytorch",
    version="0.1.0",
    packages=find_packages(),
    package_data={"ddpo_pytorch": ["assets/*.txt"]},
)
