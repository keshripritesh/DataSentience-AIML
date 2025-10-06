"""
Setup script for PersonaBot
Advanced conversational AI with dynamic personality adaptation
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
    name="personabot",
    version="1.0.0",
    author="PersonaBot Team",
    author_email="contact@personabot.ai",
    description="Advanced conversational AI with dynamic personality adaptation using Reinforcement Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/personabot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.7.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "web": [
            "streamlit>=1.22.0",
            "plotly>=5.10.0",
            "altair>=4.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "personabot=ui.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="ai chatbot conversational reinforcement-learning personality nlp",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/personabot/issues",
        "Source": "https://github.com/yourusername/personabot",
        "Documentation": "https://github.com/yourusername/personabot#readme",
    },
)
