from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = 'Placeholder'
LONG_DESCRIPTION = 'Placeholder'

# Setting up
setup(
    name="htes",
    version=VERSION,
    author="Robert Stanton",
    author_email="<stantor@clarkson.edu>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['ase'],
    keywords=['python', 'electronic structure', 'DFT', 'semi-empirical'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
