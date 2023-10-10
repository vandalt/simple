(installation)=

# Installation Guide

## Using pip

The easiest way to install the most recent stable version of simple
is with [pip](https://pip.pypa.io):

```bash
python -m pip install simple
```

## From source

Alternatively, you can get the source:

```bash
git clone https://github.com/vandalt/simple
cd simple
python -m pip install -e .
```

To install with optional dependencies, use the `test` and/or `dev` options:

```bash
python -m pip install -e ".[test,dev]"
```

## Tests

If you installed from source with the `test` or `dev` options, you can run the unit tests.
From the root of the source directory, run:

```bash
python -m pip install nox
python -m nox -s tests
```
