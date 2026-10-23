from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


PROJECT_ROOT = Path(__file__).resolve().parent
class PSSolverBuildPy(build_py):
    """Keep repository-only qualification helpers out of built wheels."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package != "pssolver.experimental":
            return modules
        return [
            item
            for item in modules
            if not item[1].startswith("stage_")
            and item[1] != "h100_shadow_qualification"
        ]


VERSION_NAMESPACE = {}
exec(
    (PROJECT_ROOT / "pssolver" / "_version.py").read_text(encoding="utf-8"),
    VERSION_NAMESPACE,
)

setup(
    name="pssolver",
    version=VERSION_NAMESPACE["__version__"],
    description=(
        "Composable tensor-product spectral solver for active-nematic "
        "simulations"
    ),
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    # Only the supported library namespace belongs in the distribution.
    # Repository-local qualification helpers under ``benchmarks/`` are not a
    # public import surface and must not shadow a consumer's own package.
    packages=find_packages(
        include=["pssolver", "pssolver.*"],
        exclude=["pssolver.control", "pssolver.control.*"],
    ),
    py_modules=["Plane_beris_edwards_stokes"],
    cmdclass={"build_py": PSSolverBuildPy},
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
