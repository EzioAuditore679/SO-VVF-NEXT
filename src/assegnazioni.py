"""Assegnazione dei mezzi ai ruoli della squadra: ottimizzazione vincolata risolta con simulated annealing.

- Variabili: i ruoli richiesti dagli incidenti, derivati dalla KB Prolog (estinzione, rinforzo,
  soccorso in quota, decontaminazione).
- Domini: i mezzi adatti al ruolo secondo la KB e raggiungibili, più SCOPERTO (si chiede un rinforzo
  a un comando vicino, al costo config.PENALITA_RUOLO_SCOPERTO).
- Vincolo rigido: un mezzo copre al più un ruolo.
- Vincolo flessibile: un incidente critico è servito quando arriva l'intera squadra, quindi per ogni
  incidente critico si paga anche il tempo di arrivo dell'ultimo mezzo (completamento della squadra),
  pesato da config.PESO_COMPLETAMENTO.
- Funzione obiettivo: tempi di percorrenza + penalità dei ruoli scoperti + completamento delle squadre critiche.
"""

import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
import modello_generativo as mg
import ricerca
from domain import TipoMezzo

INF = math.inf
SCOPERTO = None


# --- Scenario ---

@dataclass
class Caso:
    mappa: ricerca.Mappa
    mezzi: dict       # id -> (TipoMezzo, cella)
    incidenti: dict   # id -> (variabili di modello_generativo, cella)


@dataclass
class Problema:
    ruoli: list       # (incidente, ruolo)
    domini: dict      # (incidente, ruolo) -> mezzi adatti e raggiungibili
    tempo: dict       # (mezzo, incidente) -> tempo di percorrenza
    penalita: float = config.PENALITA_RUOLO_SCOPERTO
    critici: frozenset = field(default_factory=frozenset)  # incidenti con gravità operativa critica
    peso_completamento: float = config.PESO_COMPLETAMENTO


@dataclass
class Soluzione:
    costo: float
    assegnazioni: list  # (mezzo, incidente, ruolo)
    scoperti: list      # (incidente, ruolo)


"""Mappa, flotta con la composizione di config e incidenti dal modello generativo."""
def genera_caso(rng: random.Random, n_incidenti: int, n_mezzi: int, variabili=None) -> Caso:
    mappa = ricerca.genera_mappa(rng)
    libere = mappa.celle_libere()
    tipi = [TipoMezzo(t) for t in config.COMPOSIZIONE_FLOTTA]
    pesi = list(config.COMPOSIZIONE_FLOTTA.values())
    mezzi = {f"m{k}": (rng.choices(tipi, pesi)[0], rng.choice(libere)) for k in range(n_mezzi)}
    genera = variabili or mg.genera_variabili
    incidenti = {f"i{k}": (genera(rng), rng.choice(libere)) for k in range(n_incidenti)}
    return Caso(mappa, mezzi, incidenti)


"""Tempo di ogni mezzo verso ogni incidente: A* con h_PD letta dal DB di pattern della mappa.

Il DB dipende solo dalla mappa senza chiusure: in esercizio si costruisce offline per tutte le celle
(ricerca.DBPattern). Qui le tabelle sono calcolate alla prima richiesta, con lo stesso risultato."""
def tempi(caso: Caso, db: ricerca.DBPattern | None = None) -> dict:
    db = db or ricerca.DBPattern(caso.mappa, precalcola=False)
    return {(m, i): ricerca.percorso(caso.mappa, partenza, arrivo, db).costo
            for i, (_, arrivo) in caso.incidenti.items() for m, (_, partenza) in caso.mezzi.items()}


"""Coppie di incidenti abbastanza vicini da propagare il pericolo."""
def adiacenze(caso: Caso) -> list[tuple]:
    ids = sorted(caso.incidenti)
    cella = {i: caso.incidenti[i][1] for i in ids}
    return [(a, b) for x, a in enumerate(ids) for b in ids[x + 1:]
            if ricerca.h_manhattan(cella[a])(cella[b]) <= config.DISTANZA_ADIACENZA]


"""Ruoli richiesti e mezzi adatti dalla KB Prolog. Senza gravita_stimata vale la regola con la velocità vera."""
def ruoli_dalla_kb(kb, caso: Caso, gravita_stimata: dict | None = None) -> dict:
    v = {i: var for i, (var, _) in caso.incidenti.items()}
    kb.carica_scenario(
        [dict(id=i, tipo=x["tipo_edificio"], piano=x["piano"], sostanze=x["sostanze_pericolose"],
              persone=x["persone"], meteo=x["meteo"]) for i, x in v.items()],
        {m: t for m, (t, _) in caso.mezzi.items()},
        {i: (x["vulnerabilita"], x["adiacenti"]) for i, x in v.items()},
        adiacenze(caso),
        {i: x["velocita_fuoco"] for i, x in v.items()},
        gravita_stimata,
    )
    return {(i, r): kb.mezzi_adatti(i, r) for i in sorted(caso.incidenti) for r in kb.squadra(i)}


"""Domini = mezzi adatti secondo la KB e raggiungibili. Un incidente è critico se la KB gli chiede un
rinforzo, cioè se la sua gravità operativa è critica."""
def problema(adatti: dict, tempo: dict, penalita: float = config.PENALITA_RUOLO_SCOPERTO,
             peso_completamento: float = config.PESO_COMPLETAMENTO) -> Problema:
    domini = {r: [m for m in mezzi if tempo.get((m, r[0]), INF) < INF] for r, mezzi in adatti.items()}
    critici = frozenset(i for i, r in adatti if r == "rinforzo")
    return Problema(sorted(adatti), domini, tempo, penalita, critici, peso_completamento)


# --- Funzione obiettivo ---

"""Violazioni del vincolo rigido: quante volte un mezzo è usato oltre la prima."""
def conflitti(a: dict) -> int:
    uso = Counter(m for m in a.values() if m is not SCOPERTO)
    return sum(n - 1 for n in uso.values())


"""Tempo di arrivo dell'ultimo mezzo assegnato all'incidente i (0 se nessun mezzo)."""
def completamento(p: Problema, a: dict, i) -> float:
    return max((p.tempo[(m, i)] for (j, _), m in a.items() if j == i and m is not SCOPERTO), default=0)


"""Tempi + penalità dei ruoli scoperti + completamento delle squadre critiche
+ config.COSTO_CONFLITTO per ogni violazione del vincolo rigido."""
def costo(p: Problema, a: dict) -> float:
    totale = sum(p.penalita if m is SCOPERTO else p.tempo[(m, r[0])] for r, m in a.items())
    totale += p.peso_completamento * sum(completamento(p, a, i) for i in p.critici)
    return totale + config.COSTO_CONFLITTO * conflitti(a)


def _soluzione(p: Problema, a: dict) -> Soluzione:
    return Soluzione(
        costo=costo(p, a),
        assegnazioni=[(m, i, r) for (i, r), m in a.items() if m is not SCOPERTO],
        scoperti=[(i, r) for (i, r), m in a.items() if m is SCOPERTO],
    )

"""Un tentativo di simulated annealing. Parte da un'assegnazione casuale o da `partenza`;
restituisce la migliore assegnazione senza conflitti visitata."""
# --- Simulated annealing con ripartenze casuali ---
def simulated_annealing(p: Problema, rng: random.Random, passi: int = config.SA_PASSI,
                        t_iniziale: float = config.SA_T_INIZIALE, t_finale: float = config.SA_T_FINALE,
                        costo_conflitto: float = config.COSTO_CONFLITTO, partenza: dict | None = None) -> Soluzione:

    valori = {r: p.domini[r] + [SCOPERTO] for r in p.ruoli}
    ruoli_di = defaultdict(list)  # ruoli di ogni incidente critico, per il termine di completamento
    for r in p.ruoli:
        if r[0] in p.critici:
            ruoli_di[r[0]].append(r)
    migliore = {r: SCOPERTO for r in p.ruoli}
    h_migliore = costo(p, migliore)
    a = dict(partenza) if partenza is not None else {r: rng.choice(valori[r]) for r in p.ruoli}
    uso = Counter(m for m in a.values() if m is not SCOPERTO)  # quante volte è usato ogni mezzo
    if all(n <= 1 for n in uso.values()) and costo(p, a) < h_migliore:
        migliore, h_migliore = dict(a), costo(p, a)
    h = costo(p, a) + (costo_conflitto - config.COSTO_CONFLITTO) * conflitti(a)

    def valore(r, m):
        return p.penalita if m is SCOPERTO else p.tempo[(m, r[0])]

    def fine_squadra(i):
        return max((p.tempo[(a[r], i)] for r in ruoli_di[i] if a[r] is not SCOPERTO), default=0)

    raffreddamento = (t_finale / t_iniziale) ** (1 / passi)
    t = t_iniziale
    for _ in range(passi):
        r = rng.choice(p.ruoli)
        vecchio, nuovo = a[r], rng.choice(valori[r])
        if nuovo == vecchio:
            t *= raffreddamento
            continue
        # Variazione del costo: cambia solo il ruolo r, quindi basta guardare i due mezzi coinvolti
        # e, se l'incidente è critico, il completamento della sua squadra.
        delta = valore(r, nuovo) - valore(r, vecchio)
        if vecchio is not SCOPERTO and uso[vecchio] > 1:
            delta -= costo_conflitto  # si libera un mezzo conteso
        if nuovo is not SCOPERTO and uso[nuovo] > 0:
            delta += costo_conflitto  # si usa un mezzo già occupato
        critico = r[0] in p.critici
        if critico:
            prima = fine_squadra(r[0])
            a[r] = nuovo
            delta += p.peso_completamento * (fine_squadra(r[0]) - prima)
            a[r] = vecchio
        if delta <= 0 or rng.random() < math.exp(-delta / t):
            a[r] = nuovo
            if vecchio is not SCOPERTO:
                uso[vecchio] -= 1
            if nuovo is not SCOPERTO:
                uso[nuovo] += 1
            h += delta
            if h < h_migliore - 1e-9 and all(n <= 1 for n in uso.values()):
                migliore, h_migliore = dict(a), h
        t *= raffreddamento
    return _soluzione(p, migliore)

"""Risoluzione con più tentativi indipendenti, si tiene il migliore. Con `da_separabile` il primo tentativo parte
dall'assegnazione ottima della sola parte separabile del costo, gli altri da
assegnazioni casuali (ripartenze casuali)."""
def risolvi(p: Problema, seme: int = config.SEME, ripartenze: int = config.SA_RIPARTENZE,
            da_separabile: bool = True, **parametri) -> Soluzione:

    rng = random.Random(seme)
    partenze = [assegnamento_separabile(p) if da_separabile else None] + [None] * (ripartenze - 1)
    return min((simulated_annealing(p, rng, partenza=x, **parametri) for x in partenze), key=lambda s: s.costo)


# --- Riferimenti esatti (solo per la valutazione) ---

"""Assegnazione ottima del solo costo separabile (tempi + penalità), con l'algoritmo ungherese:
una colonna "scoperto" per ruolo. Ignora il completamento delle squadre critiche, quindi è ottima
solo quando non ci sono incidenti critici; utile a misurare quanto quel termine cambia la soluzione."""
def assegnamento_separabile(p: Problema) -> dict:
    if not p.ruoli:
        return {}
    mezzi = sorted({m for d in p.domini.values() for m in d})
    c = np.full((len(p.ruoli), len(mezzi) + len(p.ruoli)), 1e9)
    for x, r in enumerate(p.ruoli):
        for y, m in enumerate(mezzi):
            if m in p.domini[r]:
                c[x, y] = p.tempo[(m, r[0])]
        c[x, len(mezzi) + x] = p.penalita
    righe, colonne = linear_sum_assignment(c)
    return {p.ruoli[x]: (mezzi[y] if y < len(mezzi) else SCOPERTO) for x, y in zip(righe, colonne)}


"""Costo minimo esatto con la programmazione lineare intera (HiGHS tramite scipy.optimize.milp)"""
def ottimo_esatto(p: Problema) -> float:
    if not p.ruoli:
        return 0.0
    x = [(r, m) for r in p.ruoli for m in p.domini[r]]
    critici = sorted(p.critici)
    n_x, n_u = len(x), len(p.ruoli)
    n = n_x + n_u + len(critici)
    idx_r = {r: k for k, r in enumerate(p.ruoli)}
    idx_z = {i: n_x + n_u + k for k, i in enumerate(critici)}
    mezzi = sorted({m for _, m in x})
    righe_z = [(k, r, m) for k, (r, m) in enumerate(x) if r[0] in idx_z]

    A = lil_matrix((n_u + len(mezzi) + len(righe_z), n))
    lb, ub = [], []
    for k, (r, _) in enumerate(x):  # ogni ruolo: un mezzo oppure scoperto
        A[idx_r[r], k] = 1
    for r, k in idx_r.items():
        A[k, n_x + k] = 1
    lb += [1] * n_u
    ub += [1] * n_u
    for y, m in enumerate(mezzi):  # ogni mezzo: al più un ruolo
        for k, (_, m2) in enumerate(x):
            if m2 == m:
                A[n_u + y, k] = 1
    lb += [0] * len(mezzi)
    ub += [1] * len(mezzi)
    for y, (k, r, m) in enumerate(righe_z):  # tempo(m, i) * x[r,m] - z[i] <= 0
        A[n_u + len(mezzi) + y, k] = p.tempo[(m, r[0])]
        A[n_u + len(mezzi) + y, idx_z[r[0]]] = -1
    lb += [-np.inf] * len(righe_z)
    ub += [0] * len(righe_z)

    c = np.array([p.tempo[(m, r[0])] for r, m in x] + [p.penalita] * n_u + [p.peso_completamento] * len(critici),
                 dtype=float)
    integrali = np.array([1] * (n_x + n_u) + [0] * len(critici))
    sup = np.array([1] * (n_x + n_u) + [np.inf] * len(critici), dtype=float)
    esito = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integrali,
                 bounds=Bounds(np.zeros(n), sup))
    if not esito.success:
        raise RuntimeError(f"programmazione lineare intera non risolta: {esito.message}")
    return float(round(esito.fun, 6))


# --- Esperimento ---

"""Distanza dall'ottimo del simulated annealing (un tentativo, ripartenze) e della soluzione
dell'algoritmo ungherese che ignora il completamento delle squadre critiche."""
def valuta(kb, seme: int = config.SEME) -> list[dict]:
    rng = random.Random(seme)
    righe = []
    for n in config.N_INCIDENTI_SCENARI:
        n_mezzi = math.ceil(n * config.N_MEZZI_PER_INCIDENTE)
        for k in range(config.N_SCENARI_PER_DIMENSIONE):
            caso = genera_caso(rng, n, n_mezzi)
            tempo = tempi(caso)
            p = problema(ruoli_dalla_kb(kb, caso), tempo)
            ottimo = ottimo_esatto(p)
            tempo_max = max((t for t in tempo.values() if t < INF), default=0)
            costi = {"ungherese (senza completamento)": costo(p, assegnamento_separabile(p))}
            for nome, ripartenze, da_sep in [("1 tentativo", 1, False),
                                             (f"{config.SA_RIPARTENZE} tentativi", config.SA_RIPARTENZE, False),
                                             (f"{config.SA_RIPARTENZE} tentativi, il primo dall'ungherese",
                                              config.SA_RIPARTENZE, True)]:
                costi[nome] = risolvi(p, seme=k, ripartenze=ripartenze, da_separabile=da_sep).costo
            for nome, c in costi.items():
                righe.append(dict(incidenti=n, scenario=k, ruoli=len(p.ruoli), critici=len(p.critici),
                                  variante=nome, costo=c, ottimo=ottimo, tempo_max=tempo_max,
                                  scarto_pct=100 * (c - ottimo) / ottimo if ottimo else 0.0))
    return righe


def riepilogo(righe: list[dict]) -> list[dict]:
    gruppi: dict = {}
    for r in righe:
        gruppi.setdefault((r["incidenti"], r["variante"]), []).append(r)
    return [dict(incidenti=n, variante=v, ruoli_media=statistics.mean(r["ruoli"] for r in rs),
                 critici_media=statistics.mean(r["critici"] for r in rs),
                 ottimi_pct=100 * sum(abs(r["costo"] - r["ottimo"]) < 1e-6 for r in rs) / len(rs),
                 scarto_pct_media=statistics.mean(r["scarto_pct"] for r in rs),
                 scarto_pct_dev=statistics.stdev(r["scarto_pct"] for r in rs))
            for (n, v), rs in sorted(gruppi.items())]


# --- Taratura dei parametri ---
# Scenari generati con un seme diverso da quello della valutazione, così i valori scelti non
# sono adattati agli scenari su cui il simulated annealing viene poi valutato. Un solo tentativo per scenario, da
# un'assegnazione casuale, per misurare il solo effetto dei parametri.

TARATURA_INCIDENTI = [4, 6, 8, 10]
TARATURA_SCENARI_PER_DIMENSIONE = 8
TARATURA_PASSI = [20_000, 50_000, 100_000]
TARATURA_T_INIZIALI = [30.0, 100.0, 300.0]
SEME_TARATURA = config.SEME + 500


def scenari_taratura(kb) -> list:
    rng = random.Random(SEME_TARATURA)
    out = []
    for n in TARATURA_INCIDENTI:
        for _ in range(TARATURA_SCENARI_PER_DIMENSIONE):
            caso = genera_caso(rng, n, math.ceil(n * config.N_MEZZI_PER_INCIDENTE))
            p = problema(ruoli_dalla_kb(kb, caso), tempi(caso))
            out.append((n, p, ottimo_esatto(p)))
    return out


def _esito(c: float, ottimo: float) -> dict:
    return dict(ottimo=abs(c - ottimo) < 1e-6, scarto_pct=100 * (c - ottimo) / ottimo if ottimo else 0.0)


def _sintesi(esiti: list) -> dict:
    return dict(ottimi_pct=100 * statistics.mean(e["ottimo"] for e in esiti),
                scarto_pct_media=statistics.mean(e["scarto_pct"] for e in esiti))

"""Quota di ottimi e scarto medio per ogni combinazione di passi e temperatura iniziale."""
def taratura(problemi: list) -> list[dict]:
    righe = []
    for passi in TARATURA_PASSI:
        for t0 in TARATURA_T_INIZIALI:
            esiti = [_esito(risolvi(p, seme=k, ripartenze=1, da_separabile=False, passi=passi,
                                    t_iniziale=t0).costo, ottimo)
                     for k, (_, p, ottimo) in enumerate(problemi)]
            righe.append(dict(passi=passi, t_iniziale=t0, **_sintesi(esiti)))
    return righe


"""Costo dei conflitti scelto (penalità + 1) contro un costo pari alla somma di tutti gli altri costi dello
scenario, cioè un vincolo rigido di fatto invalicabile."""
def taratura_costo_conflitti(problemi: list) -> list[dict]:
    righe = []
    for nome in ["penalità + 1", "somma di tutti gli altri costi"]:
        esiti = []
        for k, (_, p, ottimo) in enumerate(problemi):
            if nome == "penalità + 1":
                c = config.COSTO_CONFLITTO
            else:
                c = sum(t for t in p.tempo.values() if t < INF) + p.penalita * len(p.ruoli)
            esiti.append(_esito(risolvi(p, seme=k, ripartenze=1, da_separabile=False, costo_conflitto=c).costo,
                                ottimo))
        righe.append(dict(costo_conflitto=nome, **_sintesi(esiti)))
    return righe


if __name__ == "__main__":
    import pandas as pd
    from prolog_engine import KnowledgeBase

    kb = KnowledgeBase()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    problemi = scenari_taratura(kb)
    tab = pd.DataFrame(taratura(problemi))
    tab.to_csv(config.RESULTS_DIR / "sa_taratura.csv", index=False)
    print("Taratura (un tentativo per scenario, seme di taratura):")
    print(tab.round(2).to_string(index=False))
    conf = pd.DataFrame(taratura_costo_conflitti(problemi))
    conf.to_csv(config.RESULTS_DIR / "sa_costo_conflitti.csv", index=False)
    print(conf.round(2).to_string(index=False))

    righe = valuta(kb)
    pd.DataFrame(righe).to_csv(config.RESULTS_DIR / "assegnazioni_scenari.csv", index=False)
    tabella = pd.DataFrame(riepilogo(righe))
    tabella.to_csv(config.RESULTS_DIR / "assegnazioni_riepilogo.csv", index=False)
    print("\nConfronto con l'ottimo esatto (scenari di valutazione):")
    print(tabella.round(2).to_string(index=False))
    print(f"Tempo di percorrenza massimo osservato: {max(r['tempo_max'] for r in righe):g}")
