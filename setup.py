import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="world_population_projection",
    author="Yvann Le Fay",
    description="Bayesian world population growth model",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.7.1",
        "pandas",
        "particles",
        "tqdm"
    ],
    long_description_content_type="text/markdown",
    keywords="hierarchical bayesian model world population model demographic un fertility modeling with uncertainty",
    license="MIT",
    license_files=("LICENSE",),
)
