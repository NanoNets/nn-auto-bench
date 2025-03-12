from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="nnautobench",
    version="0.1.0",
    author=["Souvik", "Ranraj", "Hozaifa", "Utkarsh"],
    author_email="benchmarks@nanonets.com",
    description="Automation benchmark for LLMs and VLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nanonets/nnautobench",
    packages=find_packages(include=["nnautobench", "nnautobench.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nnautobench=nnautobench.__main__:main",
        ],
    },
    include_package_data=True,
)
