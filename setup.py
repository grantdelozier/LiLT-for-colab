#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="LiLTfinetune",
    version="1.1",
    author="Deep Learning and Vision Computing Lab, SCUT - Modified by Grant DeLozier",
    url="https://github.com/grantdelozier/LiLT-for-colab",
    packages=find_packages(),
    python_requires=">=3.7",
    extras_require={"dev": ["flake8", "isort", "black"]},
)