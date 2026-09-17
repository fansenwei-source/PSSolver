from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent
VERSION_NAMESPACE = {}
exec(
    (PROJECT_ROOT / "pssolver" / "_version.py").read_text(encoding="utf-8"),
    VERSION_NAMESPACE,
)

setup(
    name="pssolver",
    version=VERSION_NAMESPACE["__version__"],
    description=(
        "Tensor-product spectral solver with a supported Plane "
        "Beris-Edwards-Stokes application"
    ),
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["Plane_beris_edwards_stokes"],
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "torch>=2.5",
        "tqdm>=4.64",
    ],
    entry_points={
        "console_scripts": [
            "pssolver-plane-beris-edwards="
            "pssolver.applications.plane_beris_edwards:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
