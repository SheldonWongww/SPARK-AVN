"""Legacy editable-install metadata for the Python 3.6 VLN-CE runtime.

Python 3.6 cannot use a recent setuptools release with PEP 621/PEP 660
support.  Its final pip/setuptools toolchain falls back to ``setup.py develop``
for editable installs, while newer environments continue to use
``pyproject.toml``.
"""

from setuptools import find_packages, setup


setup(
    name="navtta-core",
    version="0.1.0",
    description="Task-agnostic test-time adaptation utilities for embodied navigation",
    python_requires=">=3.6",
    packages=find_packages(where="."),
)
