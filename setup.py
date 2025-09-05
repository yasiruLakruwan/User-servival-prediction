from setuptools import setup,find_packages

## Setup package file........

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="User servival prediction",
    author="Yasiru",
    version="0.0.1",
    packages = find_packages(),
    install_requires = requirements
)