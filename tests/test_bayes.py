"""Test della rete bayesiana: validità delle CPT, Laplace, inferenza, recupero."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import bayes_engine as be
import config
import generate_dataset as gd
import modello_generativo as mg
from domain import Meteo, Persone, SostanzePericolose, TipoEdificio, VelocitaFuoco

N_GRANDE = 20_000
TOLLERANZA = 0.03  # scarto massimo ammesso tra CPT appresa e probabilità del generatore


@pytest.fixture(scope="module")
def df_piccolo() -> pd.DataFrame:
    return pd.DataFrame(gd.genera(config.N_TESTI_SINTETICI, config.SEME))


@pytest.fixture(scope="module")
def rete(df_piccolo):
    return be.stima(df_piccolo)


@pytest.fixture(scope="module")
def df_grande() -> pd.DataFrame:
    return pd.DataFrame(gd.genera(N_GRANDE, config.SEME + 1))


@pytest.fixture(scope="module")
def rete_grande(df_grande):
    return be.stima(df_grande)


# --- Struttura e CPT  ---
def test_struttura(rete):
    assert set(rete.nodes()) == set(be.STATI)
    assert set(rete.edges()) == set(be.ARCHI)
    assert rete.check_model()


def test_cpt_normalizzate(rete):
    for nodo in be.STATI:
        cpd = rete.get_cpds(nodo)
        valori = cpd.get_values()  # righe = stati del nodo, colonne = configurazioni dei genitori
        assert valori.shape[0] == len(be.STATI[nodo])
        np.testing.assert_allclose(valori.sum(axis=0), 1.0, atol=1e-9)


def test_laplace_nessuna_probabilita_nulla(rete):
    """Con pseudoconteggi positivi nessuna cella è zero, nemmeno per configurazioni mai viste."""
    for nodo in be.STATI:
        assert rete.get_cpds(nodo).get_values().min() > 0, nodo
    # Piano 0 per un edificio residenziale non compare mai nei dati: la cella resta piccola ma positiva.
    p = be.probabilita(rete, "piano", 0, tipo_edificio="residenziale")
    assert 0 < p < 0.05


def test_alpha_maggiore_appiattisce(df_piccolo):
    """Più pseudoconteggi spingono le CPT verso l'uniforme: la cella impossibile cresce."""
    p_1 = be.probabilita(be.stima(df_piccolo, alpha=1.0), "piano", 0, tipo_edificio="residenziale")
    p_50 = be.probabilita(be.stima(df_piccolo, alpha=50.0), "piano", 0, tipo_edificio="residenziale")
    assert p_50 > p_1


# --- Inferenza ---

def _p_critico(rete, **evidenza) -> float:
    base = dict(tipo_edificio="residenziale", piano="3", sostanze_pericolose="no", meteo="calmo",
                persone="assenti")
    base.update(evidenza)
    return float(be.p_critico_righe(rete, pd.DataFrame([base]))[0])


def test_p_critico_in_intervallo(rete, df_piccolo):
    p = be.p_critico_righe(rete, df_piccolo.head(50))
    assert p.shape == (50,)
    assert np.all((p >= 0) & (p <= 1))


def test_persone_confermate_aumentano_p_critico(rete):
    assert _p_critico(rete, persone="confermate") > _p_critico(rete, persone="assenti")


def test_sostanze_e_vento_aumentano_p_critico(rete):
    calmo = _p_critico(rete)
    assert _p_critico(rete, sostanze_pericolose="si") > calmo
    assert _p_critico(rete, meteo="vento_forte") > calmo


def test_fuori_fold_forma(df_piccolo):
    df = df_piccolo.head(200)
    p = be.p_critico_fuori_fold(df, df, k=3)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))


# --- Recupero delle distribuzioni del generatore ---

def test_recupero_velocita(rete_grande):
    """P(velocità | meteo, sostanze): tutte e 4 le configurazioni sono ben popolate."""
    for meteo in Meteo:
        for sost in SostanzePericolose:
            vera = mg.P_VELOCITA[(meteo, sost)]
            for vel in VelocitaFuoco:
                stimata = be.probabilita(rete_grande, "velocita_fuoco", vel.value,
                                         meteo=meteo.value, sostanze_pericolose=sost.value)
                assert abs(stimata - vera[vel]) < TOLLERANZA, (meteo, sost, vel)


def test_recupero_persone(rete_grande, df_grande):
    """P(persone | tipo, piano) sulle configurazioni con almeno 500 righe."""
    conteggi = df_grande.groupby(["tipo_edificio", "piano"]).size()
    verificate = 0
    for (tipo, piano), n in conteggi.items():
        if n < 500:
            continue
        vera = mg._distribuzione_persone(TipoEdificio(tipo), int(piano))
        for pers in Persone:
            stimata = be.probabilita(rete_grande, "persone", pers.value, tipo_edificio=tipo, piano=piano)
            assert abs(stimata - vera[pers]) < TOLLERANZA, (tipo, piano, pers)
        verificate += 1
    assert verificate >= 3


def test_recupero_meteo_e_tipo(rete_grande):
    assert abs(be.probabilita(rete_grande, "meteo", "vento_forte") - mg.P_VENTO_FORTE) < TOLLERANZA
    for tipo in TipoEdificio:
        assert abs(be.probabilita(rete_grande, "tipo_edificio", tipo.value) - mg.P_TIPO[tipo]) < TOLLERANZA


def test_campo_non_riconosciuto_e_marginalizzato(rete):
    """Un campo fuori dominio ("nessuno": non riconosciuto dal testo) non è evidenza: la query equivale a
    quella senza la variabile, invece di fallire."""
    base = dict(tipo_edificio="residenziale", piano="5", sostanze_pericolose="no", meteo="vento_forte")
    p_ignoto = be.p_critico_righe(rete, pd.DataFrame([{**base, "persone": "nessuno"}]))[0]
    p_conf = be.p_critico_righe(rete, pd.DataFrame([{**base, "persone": "confermate"}]))[0]
    p_ass = be.p_critico_righe(rete, pd.DataFrame([{**base, "persone": "assenti"}]))[0]
    assert p_ass < p_ignoto < p_conf
