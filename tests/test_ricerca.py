"""Test della ricerca del percorso: euristiche ammissibili e consistenti, PD corretta, algoritmi ottimi."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import ricerca as rc


def mappa_casuale(rng, p_chiusure=0.15, lente=True, lato=10):
    celle = [(r, k) for r in range(lato) for k in range(lato)]
    ostacoli = frozenset(c for c in celle if rng.random() < 0.2)
    libere = [c for c in celle if c not in ostacoli]
    return rc.Mappa(lato, lato, ostacoli,
                    frozenset(c for c in libere if lente and rng.random() < 0.2),
                    frozenset(c for c in libere if rng.random() < p_chiusure))


@pytest.mark.parametrize("seme", range(20))
def test_euristiche_ammissibili_consistenti_e_ordinate(seme):
    rng = random.Random(seme)
    mappa = mappa_casuale(rng)
    obiettivo = rng.choice(mappa.celle_libere())
    esatto, _ = rc.cost_to_goal(mappa, obiettivo, con_chiusure=True)  # h* sulla mappa reale
    tabella, _ = rc.cost_to_goal(mappa, obiettivo)                    # problema semplificato
    h_m, h_p = rc.h_manhattan(obiettivo), rc.h_pd(tabella)
    for c in mappa.celle_libere():
        if c in esatto:
            assert h_m(c) <= h_p(c) <= esatto[c]          # ammissibili, PD più informata
        for v, costo in mappa.successori(c):
            assert h_m(c) <= costo + h_m(v)               # consistenza (monotonicità) su ogni arco
            if h_p(v) < rc.INF:
                assert h_p(c) <= costo + h_p(v)


def test_pd_senza_chiusure_e_euristica_perfetta():
    rng = random.Random(1)
    mappa = mappa_casuale(rng, p_chiusure=0)
    obiettivo = rng.choice(mappa.celle_libere())
    tabella, _ = rc.cost_to_goal(mappa, obiettivo)
    for partenza in mappa.celle_libere():
        r = rc.a_star(partenza, lambda c: c == obiettivo, mappa.successori, rc.h_pd(tabella))
        assert r.costo == tabella.get(partenza, rc.INF)


@pytest.mark.parametrize("seme", range(20))
def test_algoritmi_ottimi_e_bfs_no(seme):
    rng = random.Random(100 + seme)
    mappa = mappa_casuale(rng)
    libere = mappa.celle_libere()
    a, b = rng.choice(libere), rng.choice(libere)
    obiettivo = lambda c: c == b
    tabella, _ = rc.cost_to_goal(mappa, b)
    ottimo = rc.a_star(a, obiettivo, mappa.successori, lambda c: 0)
    for h in (rc.h_manhattan(b), rc.h_pd(tabella)):
        assert rc.a_star(a, obiettivo, mappa.successori, h).costo == ottimo.costo
    assert rc.bfs(a, obiettivo, mappa.successori).costo >= ottimo.costo


def test_bfs_ottima_con_costi_uniformi():
    rng = random.Random(7)
    for _ in range(20):
        mappa = mappa_casuale(rng, lente=False)
        libere = mappa.celle_libere()
        a, b = rng.choice(libere), rng.choice(libere)
        obiettivo = lambda c: c == b
        assert rc.bfs(a, obiettivo, mappa.successori).costo == rc.a_star(a, obiettivo, mappa.successori,
                                                                          lambda c: 0).costo


def test_cella_lenta_costa_di_piu_e_si_aggira():
    # Corridoio di due righe: passare dalla cella lenta (0,1) costa 3, aggirarla dalla riga sotto costa 4.
    mappa = rc.Mappa(2, 3, frozenset(), frozenset({(0, 1)}), frozenset())
    obiettivo = lambda c: c == (0, 2)
    assert rc.a_star((0, 0), obiettivo, mappa.successori, lambda c: 0).costo == 4
    assert rc.bfs((0, 0), obiettivo, mappa.successori).costo == 4  # due passi, ma uno lento: 3 + 1


def test_chiusura_che_isola_l_obiettivo():
    # La PD non vede la chiusura: h è finita, ma A* esaurisce la frontiera e conclude che non c'è percorso.
    mappa = rc.Mappa(1, 3, frozenset(), frozenset(), frozenset({(0, 1)}))
    db = rc.DBPattern(mappa)
    assert db.tabella((0, 2))[(0, 0)] == 2
    r = rc.percorso(mappa, (0, 0), (0, 2), db)
    assert r.costo == rc.INF and r.completo


def test_db_di_pattern_offline_una_tabella_per_cella():
    """Il DB offline ha una tabella per ogni cella libera senza chiusure, uguale a cost_to_goal; il DB
    costruito su richiesta dà gli stessi percorsi."""
    rng = random.Random(3)
    mappa = mappa_casuale(rng)
    db = rc.DBPattern(mappa)
    assert len(db) == len(mappa.celle_libere(con_chiusure=False))
    libere = mappa.celle_libere()
    pigro = rc.DBPattern(mappa, precalcola=False)
    for _ in range(10):
        a, b = rng.choice(libere), rng.choice(libere)
        assert db.tabella(b) == rc.cost_to_goal(mappa, b)[0]
        assert rc.percorso(mappa, a, b, db).costo == rc.percorso(mappa, a, b, pigro).costo
