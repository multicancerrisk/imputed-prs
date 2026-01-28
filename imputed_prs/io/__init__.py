"""I/O module for reading and writing files."""

from imputed_prs.io.prs_loader import (
    load_prs_from_dataframe,
    load_prs_from_file,
)

__all__ = [
    "load_prs_from_dataframe",
    "load_prs_from_file",
]
