"""Generatore di chiamate al 115 sintetiche con valori veri dei campi.

Il testo è generato solo dai campi osservabili (tipo, piano, sostanze, persone, meteo),
con rumore del parlato. Le variabili nascoste non compaiono nel testo.
"""

import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from domain import (
    PAROLE_CHIAVE,
    Campo,
    Meteo,
    Persone,
    SostanzePericolose,
    TipoEdificio,
)
from modello_generativo import genera_variabili, gravita_osservata, regola_gravita

ORDINALI = {1: "primo", 2: "secondo", 3: "terzo", 4: "quarto", 5: "quinto", 6: "sesto"}

APERTURE = [
    "Pronto, c'è",
    "Salve, abbiamo",
    "Qui c'è",
    "Sono sul posto, c'è",
    "Segnalo",
    "Buongiorno, c'è",
]

FRASI_METEO = {
    Meteo.VENTO_FORTE: ["c'è vento forte", "il vento è forte"],
    Meteo.CALMO: ["non c'è vento", "il tempo è calmo"],
}

RIEMPITIVI = ["eh", "ehm", "cioè", "allora"]

CAMPI_OSSERVABILI = ["tipo_edificio", "piano", "sostanze_pericolose", "persone", "meteo"]


def _frase_edificio(rng: random.Random, edificio: TipoEdificio, piano: int) -> str:
    kw = rng.choice(PAROLE_CHIAVE[Campo.TIPO_EDIFICIO][edificio])
    if edificio is TipoEdificio.BOSCO:
        return f"un incendio in {kw}"
    return f"un incendio in {kw} al {ORDINALI[piano]} piano"


def _frase_sostanze(rng: random.Random, sostanze: SostanzePericolose) -> str:
    kw = rng.choice(PAROLE_CHIAVE[Campo.SOSTANZE_PERICOLOSE][sostanze])
    if sostanze is SostanzePericolose.SI:
        return f"vicino alle fiamme ci sono {kw}"
    return kw


def _frase_persone(rng: random.Random, persone: Persone) -> str:
    kw = rng.choice(PAROLE_CHIAVE[Campo.PERSONE][persone])
    return f"mi dicono che {kw}"


"""Imperfezioni del parlato, ciascuna con probabilità fissa."""
def _rumore(rng: random.Random, testo: str) -> str:
    if rng.random() < 0.3:
        testo = testo.lower()
    if rng.random() < 0.3:
        m = re.search(r"\. (\w)", testo)
        if m:
            riempitivo = rng.choice(RIEMPITIVI)
            testo = testo[: m.start()] + f" {riempitivo}, " + m.group(1).lower() + testo[m.end():]
    if rng.random() < 0.2:
        testo = testo.replace("piano", "p.")
    if rng.random() < 0.3:
        testo = testo.rstrip(".")
    return testo


def genera_record(rng: random.Random, idx: int) -> dict:
    v = genera_variabili(rng)
    gravita = gravita_osservata(rng, regola_gravita(v))

    apertura = rng.choice(APERTURE)
    frasi = [
        f"{apertura} {_frase_edificio(rng, v['tipo_edificio'], v['piano'])}",
        _frase_sostanze(rng, v["sostanze_pericolose"]).capitalize(),
        _frase_persone(rng, v["persone"]).capitalize(),
        rng.choice(FRASI_METEO[v["meteo"]]).capitalize(),
    ]
    testo = _rumore(rng, ". ".join(frasi) + ".")

    return {
        "id": f"c{idx:05d}",
        "testo": testo,
        "tipo_edificio": v["tipo_edificio"].value,
        "piano": v["piano"],
        "sostanze_pericolose": v["sostanze_pericolose"].value,
        "persone": v["persone"].value,
        "meteo": v["meteo"].value,
        "velocita_fuoco": v["velocita_fuoco"].value,
        "vulnerabilita": v["vulnerabilita"].value,
        "adiacenti": v["adiacenti"],
        "gravita": gravita.value,
    }


def genera(n: int, seme: int) -> list[dict]:
    rng = random.Random(seme)
    return [genera_record(rng, i) for i in range(n)]


def scrivi_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


"""Conteggio dei valori per campo, per verificare che ogni etichetta compaia."""
def copertura(records: list[dict]) -> dict[str, Counter]:
    campi = CAMPI_OSSERVABILI + ["velocita_fuoco", "vulnerabilita", "adiacenti", "gravita"]
    return {c: Counter(r[c] for r in records) for c in campi}


"""Controlla che il modello produca dipendenze reali, non campioni indipendenti."""
def verifica_dipendenze(records: list[dict]) -> None:
    print("\nP(sostanze = si | tipo):")
    for tipo in sorted({r["tipo_edificio"] for r in records}):
        sub = [r for r in records if r["tipo_edificio"] == tipo]
        p = sum(r["sostanze_pericolose"] == "si" for r in sub) / len(sub)
        print(f"  {tipo}: {p:.2f}")
    print("P(gravita = critico | persone):")
    for pers in sorted({r["persone"] for r in records}):
        sub = [r for r in records if r["persone"] == pers]
        p = sum(r["gravita"] == "critico" for r in sub) / len(sub)
        print(f"  {pers}: {p:.2f}")


if __name__ == "__main__":
    records = genera(config.N_TESTI_SINTETICI, config.SEME)
    out = config.DATA_DIR / "dataset_sintetico.csv"
    scrivi_csv(records, out)
    print(f"Scritti {len(records)} record in {out}")
    for campo, conteggi in copertura(records).items():
        print(f"  {campo}: {dict(conteggi)}")
    verifica_dipendenze(records)
