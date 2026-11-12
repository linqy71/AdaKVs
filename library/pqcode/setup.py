from setuptools import find_packages, setup


setup(
    name="pqcode",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch>=1.12", "numpy>=1.17"],
)
