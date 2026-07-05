"""Sample-free ("sufficient statistics") training kernels for scaling.

This package contracts the sample dimension into local Gram blocks so that
per-variant / per-region linear models can be fit without ever materializing the
full reference dosage matrix. See ``compute/gram_solve.py`` for the sample-free
solver (Phase 2) and ``compute/sufficient_stats.py`` for the streaming driver.
"""
