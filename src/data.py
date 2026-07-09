import pandas as pd
from pycox.datasets import metabric, support, flchain


DATASETS = {
    "METABRIC": metabric.read_df,
    "SUPPORT": support.read_df,
    "FLCHAIN": flchain.read_df,
}


def standardize_survival_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Standardise toutes les bases sous la forme :
    - duration : temps observé Y_i
    - event : indicateur Delta_i
    """
    df = df.copy()

    duration_candidates = [
        "duration", "durations", "time", "futime", "survival_time", "tte"
    ]

    event_candidates = [
        "event", "events", "death", "status", "event_observed", "label"
    ]

    duration_col = None
    event_col = None

    for col in duration_candidates:
        if col in df.columns:
            duration_col = col
            break

    for col in event_candidates:
        if col in df.columns:
            event_col = col
            break

    if duration_col is None:
        raise ValueError(
            f"Colonne de temps introuvable pour {name}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    if event_col is None:
        raise ValueError(
            f"Colonne d'événement introuvable pour {name}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    df = df.rename(columns={
        duration_col: "duration",
        event_col: "event"
    })

    df["duration"] = df["duration"].astype(float)
    df["event"] = (df["event"].astype(float) > 0).astype(int)

    df = df.dropna(subset=["duration", "event"])
    df = df[df["duration"] > 0].reset_index(drop=True)

    return df


def load_dataset(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError(f"Dataset inconnu : {name}. Choisir parmi {list(DATASETS.keys())}")

    df = DATASETS[name]()
    df = standardize_survival_columns(df, name)

    return df


def load_all_datasets() -> dict:
    return {name: load_dataset(name) for name in DATASETS.keys()}
