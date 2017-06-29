import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyplearnr",
    version = "1.0.3",
    author = "Christopher Shymansky",
    author_email = "CMShymansky@gmail.com",
    description = ("Pyplearnr is a tool designed to easily and more " \
                   "elegantly build, validate (nested k-fold cross-validation" \
                   "), and test scikit-learn pipelines."),
    license = "ALv2",
    keywords = "scikit-learn pipeline k-fold cross-validation model selection",
    url = "http://packages.python.org/pyplearnr",
    packages=['pyplearnr', 'test'],
    long_description=read('README.md'),
    install_requires=[
            'pandas',
            'numpy',
            'sklearn',
            'matplotlib',
            'pylab'
        ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: ALv2",
    ],
)
