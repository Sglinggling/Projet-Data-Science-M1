"""
Shared I/O utilities.
"""
import pandas as pd


def load_off(path: str, usecols: list[str], nrows: int | None = None) -> pd.DataFrame:
    """
    Load a subset of the Open Food Facts TSV dump without reading the full file
    into RAM.

    Parameters
    ----------
    path : str
        Path to the TSV file (en.openfoodfacts.org.products.csv or .gz).
    usecols : list[str]
        Exact column names to keep.  Only these are decoded; everything else is
        skipped by the C parser, so memory use stays proportional to usecols.
    nrows : int | None
        If given, stop after this many rows (useful for quick experiments).

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
        on_bad_lines="skip",
    )
