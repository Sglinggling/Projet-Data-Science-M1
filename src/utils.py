
import pandas as pd


def load_off(path: str, usecols: list[str], nrows: int | None = None) -> pd.DataFrame:
    """Charge le fichier TSV Open Food Facts en ne lisant que les colonnes utiles."""
    return pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
        on_bad_lines="skip",
    )
