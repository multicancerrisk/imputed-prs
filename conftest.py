"""Root conftest.

Its only job is to exist: pytest's default "prepend" import mode inserts the directory
of each ``conftest.py`` onto ``sys.path``. Placing one at the repository root therefore
makes the top-level ``benchmarks`` package importable from the test suite (the library
lives in ``imputed_prs/`` and is already importable; ``benchmarks/`` is a sibling that
would otherwise not be on the path). ``tests/conftest.py`` and its fixtures are
unaffected.
"""
