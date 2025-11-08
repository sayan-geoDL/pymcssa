from setuptools import setup, find_packages

setup(
    name="pymcssa",
    version="0.1.0",
    description="A Monte Carlo Singular Spectrum Analysis (MCSSA) package for time series analysis.",
    author="Sayan Jana",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26,<2.0",
        "pytest>=7.4,<8.0"
    ],
    packages=find_packages(where="pymcssa"),
    package_dir={"": "pymcssa"},
    include_package_data=True,
    license="MIT",
    url="https://github.com/sayan-geoDL/pymcssa",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
