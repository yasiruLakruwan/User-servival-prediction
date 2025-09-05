from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="User servival prediction",
    version="0.0.1",
    author="Yasiru",
    packages=find_packages(),
    install_requires = requirements
)

