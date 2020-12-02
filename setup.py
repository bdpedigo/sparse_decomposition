#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

readme = open("README.rst").read()
doclink = """
Documentation
-------------

The full documentation is at http://sparse_decomposition.rtfd.org."""
history = open("HISTORY.rst").read().replace(".. :changelog:", "")

setup(
    name="sparse_decomposition",
    version="0.1.0",
    description="Implements algorithms from 'A New Basis for Sparse PCA'.",
    long_description=readme + "\n\n" + doclink + "\n\n" + history,
    author="Benjamin Pedigo",
    author_email="benjamindpedigo@gmail.com",
    url="https://github.com/bdpedigo/sparse_decomposition",
    packages=["sparse_decomposition",],
    package_dir={"sparse_decomposition": "sparse_decomposition"},
    include_package_data=True,
    install_requires=[],
    license="MIT",
    zip_safe=False,
    keywords="sparse_decomposition",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
