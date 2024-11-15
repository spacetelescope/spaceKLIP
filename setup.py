#!/usr/bin/env python
# Minimal setup.py; most config information is now in pyproject.toml
import os
from setuptools import setup, find_packages

upload_to_pypi = os.environ.get("UPLOAD_TO_PYPI") == '1'

setup(packages=find_packages(),
      use_scm_version={
            "root": ".",  # Root directory of the project
            "relative_to": __file__,
            # Strip the local version identifier only if uploading to PyPI
            "local_scheme": lambda version: "" if upload_to_pypi else version.local_scheme(version),
      },
      long_description=open('README.rst').read(),
      setup_requires=['setuptools_scm'],
      )
