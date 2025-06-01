from setuptools import setup, find_packages

setup(
    name="indog2p",
    version="0.1",
    packages=find_packages(),
    install_requires=[r.strip() for r in open("requirements.txt")],
)
