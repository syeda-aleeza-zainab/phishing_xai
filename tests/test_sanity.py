def test_environment_sanity():
    import sys
    assert sys.version_info[:2] == (3, 10), "Python version must be 3.10"
