import simple


def test_simple_version():
    from importlib.metadata import version

    assert simple.__version__ == version("simple")
