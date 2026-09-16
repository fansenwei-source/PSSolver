from setuptools import find_packages, setup

setup(
    name="pssolver",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pssolver-plane-beris-edwards="
            "pssolver.applications.plane_beris_edwards:main",
        ],
    },
)
