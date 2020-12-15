#!/usr/bin/env python

from distutils.core import setup

setup(
    name="lpg-jax",
    version="0.0.1",
    description="A jax implementation of the Learned Policy Gradient algorithm",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignateli/lpg",
    packages=["lpg", "lpg.environments"],
    install_requires=open("requirements.txt", "r").readlines(),
)