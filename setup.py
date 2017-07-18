import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyplearnr",
    version = "1.0.11",
    author = "Christopher Shymansky",
    author_email = "CMShymansky@gmail.com",
    description = ("Pyplearnr is a tool designed to easily and more " \
                   "elegantly build, validate (nested k-fold cross-validation" \
                   "), and test scikit-learn pipelines."),
    license = "OSI Approved :: Apache Software License",
    keywords = "scikit-learn pipeline k-fold cross-validation model selection",
    url = "http://packages.python.org/pyplearnr",
    packages=['pyplearnr', 'test'],
    long_description=read('README.md'),
    install_requires=[
            'pandas',
            'numpy',
            'sklearn',
            'matplotlib'
        ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
)
