"""Test dell'adattatore di ingresso (src/nlp_extract.py).

Essi  verificano che ogni regola faccia quello che dichiara, su frasi scritte in modo diverso dai template del generatore
(sinonimi, negazioni, incertezza, abbreviazioni), e che tutte le chiamate generate siano lette.
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import generate_dataset as gd
from nlp_extract import estrai_campi


def campo(testo: str, nome: str):
    return estrai_campi(testo)[nome]


@pytest.mark.parametrize("testo, atteso", [
    ("Brucia una villetta in periferia", "residenziale"),
    ("Fumo da un'abitazione in centro", "residenziale"),
    ("Incendio in un'officina meccanica", "industriale"),
    ("Fiamme in un magazzino di legname", "industriale"),
    ("Stanno bruciando delle sterpaglie lungo la strada", "bosco"),
    ("Un'auto in fiamme sul raccordo", None),  # nessun edificio riconosciuto
])
def test_tipo_di_edificio(testo, atteso):
    assert campo(testo, "tipo_edificio") == atteso


@pytest.mark.parametrize("testo, atteso", [
    ("Incendio in un condominio al terzo piano", 3),
    ("Fumo da un appartamento al 4° piano", 4),
    ("Incendio in un palazzo al 2 p.", 2),  # abbreviazione del parlato
    ("Brucia una pineta", 0),               # nessun piano indicato
])
def test_piano(testo, atteso):
    assert campo(testo, "piano") == atteso


@pytest.mark.parametrize("testo, atteso", [
    ("Nel garage c'è una tanica di benzina.", "si"),
    ("Accanto al capannone ci sono bombole di GPL.", "si"),
    ("Non ci sono bombole nell'edificio.", "no"),         # negazione nella stessa frase
    ("Nessuna sostanza pericolosa presente.", "no"),
    ("Brucia un appartamento al primo piano.", "no"),     # niente di menzionato: default
])
def test_sostanze_pericolose(testo, atteso):
    assert campo(testo, "sostanze_pericolose") == atteso


@pytest.mark.parametrize("testo, atteso", [
    ("Una signora è bloccata in bagno.", "confermate"),
    ("Due ragazzi non riescono a scendere.", "confermate"),
    ("Forse c'è qualcuno al piano di sopra.", "incerte"),  # l'incertezza prevale
    ("Non si sa se l'inquilino sia in casa.", "incerte"),
    ("Nessuno è rimasto intrappolato.", "assenti"),         # conferma negata
    ("Sono tutti fuori in strada.", "assenti"),
    ("Brucia un deposito.", None),                          # persone non menzionate
])
def test_persone(testo, atteso):
    assert campo(testo, "persone") == atteso


@pytest.mark.xfail(reason="limite noto: 'non ci sono persone dentro' non è tra le forme di assenza", strict=True)
def test_persone_limite_noto():
    assert campo("Non ci sono persone dentro.", "persone") == "assenti"


@pytest.mark.parametrize("testo, atteso", [
    ("Il vento è sostenuto.", "vento_forte"),
    ("Non tira vento.", "calmo"),  # la calma va riconosciuta prima della parola "vento"
    ("Aria ferma, tempo stabile.", "calmo"),
    ("Incendio in una casa.", None),
])
def test_meteo(testo, atteso):
    assert campo(testo, "meteo") == atteso


def test_chiamata_completa_in_linguaggio_libero():
    testo = ("Fiamme in una casa colonica, al primo piano c'è una bombola di gas. "
             "Una persona è bloccata in camera. Vento forte da est.")
    assert estrai_campi(testo) == dict(tipo_edificio="residenziale", piano=1, sostanze_pericolose="si",
                                       persone="confermate", meteo="vento_forte")


def test_legge_tutte_le_chiamate_del_generatore():
    """Sulle chiamate generate (template più rumore del parlato) i campi coincidono con quelli veri."""
    rng = random.Random(7)
    for k in range(500):
        r = gd.genera_record(rng, k)
        estratti = estrai_campi(r["testo"])
        for nome in ["tipo_edificio", "piano", "sostanze_pericolose", "persone", "meteo"]:
            assert estratti[nome] == r[nome], (nome, r["testo"])
