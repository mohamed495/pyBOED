"""
pyBOED - Bayesian Optimal Experimental Design with Model Reduction
===================================================================

A comprehensive Python library for Bayesian Optimal Experimental Design (BOED)
with integrated dimensionality reduction methods.

Combines:
- BOED: Optimal experimental design for PDE-constrained inverse problems
- Dimensionality Reduction: PCA, KLE, POD, RB, Active Subspaces, LIS
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyBOED",
    version="2.0.0",
    author="M. Doumbou",
    author_email="your.email@example.com",
    description="Bayesian Optimal Experimental Design with Model Reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyBOED",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.5",
        "plotly>=5.0",
        "pyyaml>=5.4",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "jupyter>=1.0",
            "notebook>=6.4",
        ],
    },
    package_data={
        "boed": ["config/*.yaml"],
    },
)
