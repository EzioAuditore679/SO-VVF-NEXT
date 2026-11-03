"""Modello generativo causale dei dati sintetici.

Variabili nascoste (non compaiono nel testo della chiamata): velocità del fuoco,
vulnerabilità della zona, numero di edifici adiacenti.

Tutte le probabilità sono assunzioni dichiarate.
"""

import random

import config
from domain import (
    Gravita,
    Meteo,
    Persone,
    SostanzePericolose,
    TipoEdificio,
    VelocitaFuoco,
    Vulnerabilita,
)

P_TIPO = {
    TipoEdificio.RESIDENZIALE: 0.40,
    TipoEdificio.INDUSTRIALE: 0.35,
    TipoEdificio.BOSCO: 0.25,
}

# Pesi per i piani 1..6. Il bosco ha piano 0.
PESI_PIANO = {
    TipoEdificio.RESIDENZIALE: [0.25, 0.25, 0.20, 0.15, 0.10, 0.05],
    TipoEdificio.INDUSTRIALE: [0.40, 0.30, 0.20, 0.10, 0.00, 0.00],
}

P_SOSTANZE_SI = {
    TipoEdificio.RESIDENZIALE: 0.30,
    TipoEdificio.INDUSTRIALE: 0.75,
    TipoEdificio.BOSCO: 0.05,
}

P_VENTO_FORTE = 0.40

P_VELOCITA = {
    (Meteo.VENTO_FORTE, SostanzePericolose.SI): {
        VelocitaFuoco.ALTA: 0.70, VelocitaFuoco.MEDIA: 0.25, VelocitaFuoco.BASSA: 0.05,
    },
    (Meteo.VENTO_FORTE, SostanzePericolose.NO): {
        VelocitaFuoco.ALTA: 0.40, VelocitaFuoco.MEDIA: 0.40, VelocitaFuoco.BASSA: 0.20,
    },
    (Meteo.CALMO, SostanzePericolose.SI): {
        VelocitaFuoco.ALTA: 0.35, VelocitaFuoco.MEDIA: 0.40, VelocitaFuoco.BASSA: 0.25,
    },
    (Meteo.CALMO, SostanzePericolose.NO): {
        VelocitaFuoco.ALTA: 0.10, VelocitaFuoco.MEDIA: 0.30, VelocitaFuoco.BASSA: 0.60,
    },
}

BASE_CONFERMATE = {
    TipoEdificio.RESIDENZIALE: 0.10,
    TipoEdificio.INDUSTRIALE: 0.15,
    TipoEdificio.BOSCO: 0.20,
}
INCREMENTO_CONFERMATE_PER_PIANO = 0.06
TETTO_CONFERMATE = 0.65
P_PERSONE_INCERTE = 0.20
P_VULNERABILITA_ALTA = 0.30
P_ADIACENTI = [0.40, 0.30, 0.20, 0.10]  # probabilità di 0, 1, 2, 3 edifici adiacenti

# Regola di gravità: punteggio sommato e soglie.
PUNTI_PERSONE = {Persone.CONFERMATE: 2, Persone.INCERTE: 1, Persone.ASSENTI: 0}
PUNTI_VELOCITA = {VelocitaFuoco.ALTA: 2, VelocitaFuoco.MEDIA: 1, VelocitaFuoco.BASSA: 0}
SOGLIA_CRITICO = 5
SOGLIA_ALTO = 3

LIVELLI_GRAVITA = [Gravita.CRITICO, Gravita.ALTO, Gravita.BASSO]


def _scegli(rng: random.Random, distribuzione: dict):
    chiavi = list(distribuzione.keys())
    pesi = list(distribuzione.values())
    return rng.choices(chiavi, weights=pesi)[0]


def _distribuzione_persone(tipo: TipoEdificio, piano: int) -> dict:
    confermate = min(BASE_CONFERMATE[tipo] + INCREMENTO_CONFERMATE_PER_PIANO * piano, TETTO_CONFERMATE)
    return {
        Persone.CONFERMATE: confermate,
        Persone.INCERTE: P_PERSONE_INCERTE,
        Persone.ASSENTI: 1.0 - confermate - P_PERSONE_INCERTE,
    }


"""Campiona una situazione completa seguendo il grafo causale."""
def genera_variabili(rng: random.Random) -> dict:
    tipo = _scegli(rng, P_TIPO)
    piano = 0 if tipo is TipoEdificio.BOSCO else rng.choices(range(1, 7), weights=PESI_PIANO[tipo])[0]
    sostanze = SostanzePericolose.SI if rng.random() < P_SOSTANZE_SI[tipo] else SostanzePericolose.NO
    meteo = Meteo.VENTO_FORTE if rng.random() < P_VENTO_FORTE else Meteo.CALMO
    velocita = _scegli(rng, P_VELOCITA[(meteo, sostanze)])
    persone = _scegli(rng, _distribuzione_persone(tipo, piano))
    vulnerabilita = Vulnerabilita.ALTA if rng.random() < P_VULNERABILITA_ALTA else Vulnerabilita.BASSA
    adiacenti = rng.choices([0, 1, 2, 3], weights=P_ADIACENTI)[0]
    return {
        "tipo_edificio": tipo,
        "piano": piano,
        "sostanze_pericolose": sostanze,
        "meteo": meteo,
        "velocita_fuoco": velocita,
        "persone": persone,
        "vulnerabilita": vulnerabilita,
        "adiacenti": adiacenti,
    }

"""Regola di dominio deterministica, senza rumore."""
def regola_gravita(v: dict) -> Gravita:
    punti = PUNTI_PERSONE[v["persone"]] + PUNTI_VELOCITA[v["velocita_fuoco"]]
    punti += 1 if v["sostanze_pericolose"] is SostanzePericolose.SI else 0
    punti += 1 if v["vulnerabilita"] is Vulnerabilita.ALTA else 0
    punti += 1 if v["adiacenti"] >= 2 else 0
    if punti >= SOGLIA_CRITICO:
        return Gravita.CRITICO
    if punti >= SOGLIA_ALTO:
        return Gravita.ALTO
    return Gravita.BASSO

"""Etichetta osservata: con probabilità config.RUMORE_GRAVITA si tenta uno spostamento di un livello.

Agli estremi (critico, basso) lo spostamento verso l'esterno è troncato e l'etichetta resta invariata."""
def gravita_osservata(rng: random.Random, regola: Gravita) -> Gravita:

    if rng.random() >= config.RUMORE_GRAVITA:
        return regola
    i = LIVELLI_GRAVITA.index(regola) + rng.choice([-1, 1])
    i = min(max(i, 0), len(LIVELLI_GRAVITA) - 1)
    return LIVELLI_GRAVITA[i]
