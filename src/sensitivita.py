"""Analisi di sensibilità al rumore delle etichette.

Gli esperimenti A, B, C, rete e oracolo ripetuti con rumore 0.05, 0.12 e 0.20.
Il recupero delle CPT rispetto al generatore è verificato in tests/test_bayes.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import config
import generate_dataset as gd
import train_models as tm

RUMORI = [0.05, 0.12, 0.20]


def sensibilita_rumore(valori: list = RUMORI) -> pd.DataFrame:
    originale = config.RUMORE_GRAVITA
    tutte = []
    try:
        for r in valori:
            config.RUMORE_GRAVITA = r
            df = pd.DataFrame(gd.genera(config.N_TESTI_SINTETICI, config.SEME))
            for riga in tm.esegui(df):
                riga["rumore"] = r
                tutte.append(riga)
    finally:
        config.RUMORE_GRAVITA = originale
    return tm.riassumi(tutte, ["rumore", "esperimento", "modello"])


if __name__ == "__main__":
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rum = sensibilita_rumore()
    rum.to_csv(config.RESULTS_DIR / "sensitivita_rumore.csv", index=False)
    print(rum.round(3).to_string(index=False))
