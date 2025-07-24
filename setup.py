#!/usr/bin/env python3
"""
Setup script for MASS (Multi-Agent Scaling System)
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mass-agent",
    version="0.0.1",
    description="Multi-Agent Scaling System - A powerful framework for collaborative AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MASS Team",
    author_email="contact@massagent.dev",
    url="https://github.com/Leezekun/MassAgent",
    project_urls={
        "Bug Reports": "https://github.com/Leezekun/MassAgent/issues",
        "Source": "https://github.com/Leezekun/MassAgent",
        "Documentation": "https://github.com/Leezekun/MassAgent/blob/main/README.md",
    },
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "mass=mass.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, multi-agent, collaboration, llm, gpt, gemini, grok, openai",
    include_package_data=True,
    package_data={
        "mass": ["examples/*.yaml", "backends/.env.example"],
    },
    zip_safe=False,
) 