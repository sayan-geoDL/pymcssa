from setuptools import setup, find_packages

setup(
    name="pymcssa",
    version="0.1.1",
    author="Sayan Jana",
    author_email="janasayan143@gmail.com",
    description="A Monte Carlo Singular Spectrum Analysis (MCSSA) package for time series analysis.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sayan-geoDL/pymcssa",
    license="MIT",
    packages=find_packages(include=["pymcssa", "pymcssa.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26,<2.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research",
    ],
    project_urls={
        "Homepage": "https://github.com/sayan-geoDL/pymcssa",
        "Documentation": "https://sayan-geoDL.github.io/pymcssa/",
        "Issues": "https://github.com/sayan-geoDL/pymcssa/issues",
    },
)

