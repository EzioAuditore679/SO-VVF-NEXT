"""Test dell'assegnazione dei mezzi (simulated annealing), con l'ottimo esatto come riferimento."""

import itertools
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import assegnazioni as a


def problema_casuale(rng, n_incidenti, n_mezzi):
    """Ruoli (1-3 per incidente), mezzi adatti casuali, tempi in [1, 6], alcuni mezzi irraggiungibili."""
    mezzi = [f"m{k}" for k in range(n_mezzi)]
    adatti, tempo = {}, {}
    for k in range(n_incidenti):
        i = f"i{k}"
        for m in mezzi:
            tempo[(m, i)] = a.INF if rng.random() < 0.1 else rng.randint(1, 6)
        for ruolo in ["estinzione", "rinforzo", "soccorso_quota"][:rng.randint(1, 3)]:
            adatti[(i, ruolo)] = [m for m in mezzi if rng.random() < 0.5]
    return a.problema(adatti, tempo, penalita=10)


def ottimo_per_enumerazione(p) -> float:
    """Costo minimo provando tutte le assegnazioni ammissibili (solo per istanze piccole)."""
    migliore = a.INF
    for valori in itertools.product(*[p.domini[r] + [a.SCOPERTO] for r in p.ruoli]):
        assegnazione = dict(zip(p.ruoli, valori))
        if a.conflitti(assegnazione) == 0:
            migliore = min(migliore, a.costo(p, assegnazione))
    return migliore


@pytest.mark.parametrize("seme", range(20))
def test_ottimo_esatto_coincide_con_l_enumerazione(seme):
    rng = random.Random(seme)
    p = problema_casuale(rng, rng.randint(1, 3), rng.randint(2, 4))
    assert a.ottimo_esatto(p) == ottimo_per_enumerazione(p)


@pytest.mark.parametrize("seme", range(30))
def test_simulated_annealing_ammissibile_e_ottimo_su_istanze_piccole(seme):
    rng = random.Random(100 + seme)
    p = problema_casuale(rng, rng.randint(2, 4), rng.randint(3, 8))
    s = a.risolvi(p, seme=seme)
    mezzi = [m for m, _, _ in s.assegnazioni]
    assert len(mezzi) == len(set(mezzi))                              # vincolo rigido rispettato
    assert all(m in p.domini[(i, r)] for m, i, r in s.assegnazioni)   # solo mezzi adatti
    assert sorted([(i, r) for _, i, r in s.assegnazioni] + s.scoperti) == p.ruoli
    assert s.costo == a.ottimo_esatto(p)


def test_conflitto_costa_piu_della_penalita():
    p = a.problema({("i0", "estinzione"): ["m0"], ("i1", "estinzione"): ["m0"]},
                   {("m0", "i0"): 2, ("m0", "i1"): 3}, penalita=10)
    con_conflitto = {("i0", "estinzione"): "m0", ("i1", "estinzione"): "m0"}
    ammissibile = {("i0", "estinzione"): "m0", ("i1", "estinzione"): a.SCOPERTO}
    assert a.conflitti(con_conflitto) == 1
    assert a.costo(p, ammissibile) < a.costo(p, con_conflitto)
    assert a.risolvi(p).costo == 12


def test_ruolo_senza_mezzi_resta_scoperto():
    p = a.problema({("i0", "soccorso_quota"): []}, {}, penalita=10)
    s = a.risolvi(p)
    assert s.costo == 10 and s.scoperti == [("i0", "soccorso_quota")]


def test_completamento_rende_il_problema_non_separabile():
    """L'ungherese minimizza solo i tempi: per l'incidente critico i0 sceglie a e b (somma 10), ma la squadra
    è completa solo all'arrivo di b (tempo 9). Con a e c la somma è maggiore ma la squadra arriva prima:
    l'ottimo vero cambia e il simulated annealing lo trova."""
    tempo = {("a", "i0"): 1, ("b", "i0"): 9, ("c", "i0"): 5, ("c", "i1"): 1, ("d", "i1"): 6}
    p = a.problema({("i0", "estinzione"): ["a", "b", "c"], ("i0", "rinforzo"): ["a", "b", "c"],
                    ("i1", "estinzione"): ["c", "d"]}, tempo, penalita=100)
    assert p.critici == {"i0"}
    separabile = a.assegnamento_separabile(p)
    assert set(separabile.values()) == {"a", "b", "c"} and a.costo(p, separabile) == 20
    assert a.ottimo_esatto(p) == 17 == ottimo_per_enumerazione(p)
    assert a.risolvi(p).costo == 17


@pytest.mark.parametrize("seme", range(10))
def test_partenza_dall_ungherese_mai_peggiore(seme):
    """Il primo tentativo parte dalla soluzione dell'ungherese e si tiene la migliore visitata,
    quindi il risultato non è mai peggiore di quella soluzione."""
    rng = random.Random(500 + seme)
    p = problema_casuale(rng, rng.randint(3, 5), rng.randint(5, 9))
    s = a.risolvi(p, seme=seme, ripartenze=1, passi=200)
    assert s.costo <= a.costo(p, a.assegnamento_separabile(p)) + 1e-9
