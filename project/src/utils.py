
import pandas as pd


def load_off(path: str, usecols: list[str], nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
        on_bad_lines="skip",
    )
