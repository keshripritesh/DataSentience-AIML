#!/usr/bin/env python3
"""
FluidNetSim: Advanced Physics-Informed Neural Fluid Dynamics Simulator
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fluidnetsim",
    version="1.0.0",
    author="FluidNetSim Team",
    author_email="contact@fluidnetsim.org",
    description="Advanced Physics-Informed Neural Fluid Dynamics Simulator",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fluidnetsim",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/fluidnetsim/issues",
        "Source": "https://github.com/yourusername/fluidnetsim",
        "Documentation": "https://fluidnetsim.readthedocs.io/",
    },
    packages=find_packages(where="."),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",  # CUDA 11.8 support
        ],
    },
    entry_points={
        "console_scripts": [
            "fluidnetsim=fluidnetsim.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="fluid dynamics, neural networks, physics-informed, simulation, CFD",
)
