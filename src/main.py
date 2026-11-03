"""SO-VVF-NEXT: demo integrata e pipeline degli esperimenti.

    python src/main.py                                  demo su quattro chiamate (pochi secondi)
    python src/main.py esperimenti                      rigenera dataset, test, risultati e grafici 
    python src/main.py esperimenti --senza-sensibilita  come sopra, senza l'analisi di sensibilità 
    python src/main.py grafici                          solo i grafici (src/plotter.py), dai CSV in results/

La demo percorre l'intera catena: testo della chiamata → campi (NLP) → gravità stimata da ML, rete
bayesiana e rete + regola della KB → squadra richiesta con le spiegazioni Prolog → assegnazione dei mezzi
(simulated annealing) → percorsi (BFS, LcFS, A*) → verifica di integrità della proposta.
Il dispatch usa la gravità della fonte rete + KB, che nella valutazione end-to-end manca meno incidenti
critici 
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import config
import plotter


""" Quattro chiamate pensate per mostrare i casi interessanti. c2 e c3 sono vicine: se c2 è critica,
 c3 (alta) diventa critica per propagazione. Il registro della zona dà vulnerabilità e adiacenti."""
CHIAMATE = [
    dict(id="c1", cella=(3, 4), vulnerabilita="bassa", adiacenti=1,
         testo="Pronto, c'è un incendio in una palazzina al quinto piano. Mi dicono che ci sono persone "
               "intrappolate. C'è vento forte."),
    dict(id="c2", cella=(10, 10), vulnerabilita="alta", adiacenti=2,
         testo="Salve, abbiamo un incendio in un capannone al primo piano. Vicino alle fiamme ci sono "
               "bombole di gas. Mi dicono che c'è qualcuno bloccato. Il vento è forte."),
    dict(id="c3", cella=(11, 12), vulnerabilita="bassa", adiacenti=1,
         testo="Segnalo un incendio in un deposito al secondo piano. Niente sostanze pericolose. "
               "Mi dicono che qualcuno non riesce a uscire. C'è vento forte."),
    dict(id="c4", cella=(17, 17), vulnerabilita="bassa", adiacenti=0,
         testo="Buongiorno, c'è un incendio in una pineta. Nessuna sostanza pericolosa. "
               "Mi dicono che tutti sono usciti. Non c'è vento."),
]

FLOTTA = {
    "aps1": ("autopompa", (2, 2)), "aps2": ("autopompa", (12, 8)), "aps3": ("autopompa", (8, 15)),
    "aps4": ("autopompa", (18, 3)), "as1": ("autoscala", (5, 1)), "nbcr1": ("nucleo_nbcr", (14, 10)),
    "cnd1": ("canadair", (19, 19)),
}

"""Mappa generata dal seme, con le celle di incidenti e stazioni libere."""
def _mappa_demo():
    import ricerca

    m = ricerca.genera_mappa(random.Random(config.SEME))
    occupate = {c["cella"] for c in CHIAMATE} | {cella for _, cella in FLOTTA.values()}
    return ricerca.Mappa(m.righe, m.colonne, m.ostacoli - occupate, m.lente, m.chiusure - occupate)


"""1. Campi estratti dal testo, più il registro della zona."""
def _passo_campi(ids: list) -> tuple[dict, pd.DataFrame]:
    import train_models as tm
    from nlp_extract import estrai_campi

    print("\n1. Campi estratti dal testo delle chiamate")
    campi = {c["id"]: estrai_campi(c["testo"]) for c in CHIAMATE}
    for c in CHIAMATE:
        print(f"  {c['id']}: {c['testo']}")
        print(f"      → {campi[c['id']]}  registro: vulnerabilità {c['vulnerabilita']}, "
              f"adiacenti {c['adiacenti']}")
    ev = tm.campi_estratti(pd.Series([c["testo"] for c in CHIAMATE]))
    ev.index = ids
    ev["vulnerabilita"] = [c["vulnerabilita"] for c in CHIAMATE]
    ev["adiacenti"] = [str(c["adiacenti"]) for c in CHIAMATE]
    return campi, ev


"""2. Gravità secondo le diverse fonti. Lascia nella KB la fonte usata per il dispatch (rete + KB)."""
def _passo_gravita(kb, stimatori: dict, ev: pd.DataFrame, carica, adiacenze: list) -> None:
    import bayes_engine as be

    ids = list(ev.index)
    ml = stimatori["ml"].predict(ev[stimatori["colonne_ml"]])
    p = be.distribuzione_gravita(stimatori["rete"], ev, be.OSSERVATE_CON_REGISTRO)
    rete = be.gravita_piu_probabile(stimatori["rete"], ev, be.OSSERVATE_CON_REGISTRO)
    vel = dict(zip(ids, be.velocita_piu_probabile(stimatori["rete"], ev, be.OSSERVATE)))

    carica({i: "alta" for i in ids})
    prudente = {i: kb.gravita(i) for i in ids}
    carica(vel)

    print("\n2. Gravità stimata dalle diverse fonti")
    tab = pd.DataFrame({
        "ML": ml, "rete": rete, "P(critico) rete": p[:, be.STATI["gravita"].index("critico")].round(2),
        "velocità (rete)": [vel[i] for i in ids], "rete + KB": [kb.gravita(i) for i in ids],
        "prudente": [prudente[i] for i in ids], "operativa (rete + KB)": [kb.gravita_operativa(i) for i in ids],
    }, index=ids)
    print(tab.to_string())
    print(f"  Incidenti adiacenti (distanza ≤ {config.DISTANZA_ADIACENZA}): {adiacenze or 'nessuno'}")
    print("  Abduzione: spiegazioni minime e coerenti della gravità stimata da ML e rete (assunzioni sulla velocità)")
    for i, g_ml, g_rete in zip(ids, ml, rete):
        singola = kb.spiegazioni([(i, g_ml)])
        entrambe = kb.spiegazioni([(i, g_ml), (i, g_rete)])
        testo = "; ".join(" ∧ ".join(h) for h in singola) or "nessuna: la stima del ML è incoerente con la regola"
        print(f"    {i}: gravità ML {g_ml} spiegata da {testo}")
        if g_ml != g_rete:
            esito = "spiegabili insieme" if entrambe else "incoerenti (richiederebbero due velocità diverse)"
            print(f"        ML ({g_ml}) e rete ({g_rete}) insieme: {esito}")


"""3. Ruoli richiesti dalla KB, con motivo e mezzi adatti."""
def _passo_squadra(kb, ids: list) -> dict:
    print("\n3. Squadra richiesta dalla KB, con le spiegazioni")
    adatti = {}
    for i in ids:
        for ruolo, motivo in kb.motivi(i):
            adatti[(i, ruolo)] = kb.mezzi_adatti(i, ruolo)
            print(f"  {i} {ruolo:16s} perché {motivo:32s} mezzi adatti: {', '.join(adatti[(i, ruolo)]) or '-'}")
    return adatti


"""4. Assegnazione dei mezzi ai ruoli con simulated annealing, confrontata con l'ottimo esatto."""
def _passo_assegnazione(p):
    import assegnazioni as asg

    print("\n4. Assegnazione dei mezzi: simulated annealing con ripartenze casuali")
    soluzione = asg.risolvi(p)
    for m, i, ruolo in sorted(soluzione.assegnazioni, key=lambda a: (a[1], a[2])):
        print(f"  {m:6s} ({FLOTTA[m][0]:12s}) → {i} {ruolo:16s} tempo {p.tempo[(m, i)]:g}")
    for i, ruolo in soluzione.scoperti:
        print(f"  ruolo scoperto: {i} {ruolo} → richiesta di rinforzi (penalità {p.penalita})")
    if p.critici:
        print(f"  Incidenti critici (si paga anche l'arrivo dell'ultimo mezzo della squadra): {', '.join(sorted(p.critici))}")
    print(f"  Costo totale {soluzione.costo:g}; ottimo esatto {asg.ottimo_esatto(p):g}")
    return soluzione


"""5. Percorso di ogni mezzo assegnato: nodi espansi da BFS, LcFS, A* Manhattan e A* PD."""
def _passo_percorsi(mappa, soluzione, tempo: dict) -> None:
    import ricerca

    print("\n5. Percorsi dei mezzi assegnati: nodi espansi da ogni algoritmo")
    print(f"  {'mezzo':6s} {'incidente':9s} {'tempo':>5s} {'BFS':>14s} {'LcFS':>6s} {'A* Manhattan':>13s} {'A* PD':>6s}")
    celle = {c["id"]: c["cella"] for c in CHIAMATE}
    db = ricerca.DBPattern(mappa)  # offline: una tabella per ogni cella libera della mappa senza chiusure
    for m, i, _ in sorted(soluzione.assegnazioni, key=lambda a: (a[1], a[0])):
        partenza, arrivo = FLOTTA[m][1], celle[i]
        tabella = db.tabella(arrivo)
        obiettivo = lambda c, arrivo=arrivo: c == arrivo
        bfs = ricerca.bfs(partenza, obiettivo, mappa.successori)
        euristiche = {"LcFS": lambda c: 0, "M": ricerca.h_manhattan(arrivo), "PD": ricerca.h_pd(tabella)}
        e = {nome: ricerca.a_star(partenza, obiettivo, mappa.successori, h).espansi for nome, h in euristiche.items()}
        bfs_testo = f"{bfs.espansi} (tempo {bfs.costo:g})"
        print(f"  {m:6s} {i:9s} {tempo[(m, i)]:>5g} {bfs_testo:>14s} {e['LcFS']:>6} {e['M']:>13} {e['PD']:>6}")
    print(f"  DB di pattern costruito offline: {len(db)} tabelle, {db.espansi_offline} nodi espansi "
          f"(mappa senza chiusure); online A* legge la tabella dell'incidente.")


def demo() -> None:
    import assegnazioni as asg
    import valutazione_dispatch as vd
    from domain import TipoMezzo
    from prolog_engine import KnowledgeBase

    pd.set_option("display.width", 160)
    df = pd.read_csv(config.DATA_DIR / "dataset_sintetico.csv")
    print("Addestramento di regressione logistica e rete bayesiana sul dataset sintetico...")
    stimatori = vd.addestra_stimatori(df)

    ids = [c["id"] for c in CHIAMATE]
    campi, ev = _passo_campi(ids)

    # Scenario nella KB: incidenti con i campi estratti, flotta, registro e adiacenze sulla mappa.
    kb = KnowledgeBase()
    mappa = _mappa_demo()
    caso = asg.Caso(mappa, {m: (TipoMezzo(t), cella) for m, (t, cella) in FLOTTA.items()},
                    {c["id"]: (None, c["cella"]) for c in CHIAMATE})
    incidenti = [dict(id=i, tipo=campi[i]["tipo_edificio"], piano=campi[i]["piano"] or 0,
                      sostanze=campi[i]["sostanze_pericolose"], persone=campi[i]["persone"],
                      meteo=campi[i]["meteo"]) for i in ids]
    registro = {c["id"]: (c["vulnerabilita"], c["adiacenti"]) for c in CHIAMATE}
    adiacenze = asg.adiacenze(caso)
    mezzi = {m: t for m, (t, _) in FLOTTA.items()}

    def carica(velocita):
        kb.carica_scenario(incidenti, mezzi, registro, adiacenze, velocita)

    _passo_gravita(kb, stimatori, ev, carica, adiacenze)
    adatti = _passo_squadra(kb, ids)
    p = asg.problema(adatti, asg.tempi(caso))
    soluzione = _passo_assegnazione(p)
    _passo_percorsi(mappa, soluzione, p.tempo)

    conflitti = kb.conflitti(soluzione.assegnazioni)
    print("\n6. Verifica di integrità in Prolog (conflitti minimi della proposta):",
          "nessuno, proposta valida" if not conflitti else conflitti)


# --- Pipeline ---

def _esegui(*argomenti: str) -> None:
    print(f"\n$ python {' '.join(argomenti)}", flush=True)
    subprocess.run([sys.executable, *argomenti], cwd=ROOT, check=True)


def esperimenti(sensibilita: bool = True) -> None:
    _esegui("src/generate_dataset.py")
    _esegui("-m", "pytest", "-q", "tests")
    _esegui("src/ricerca.py")
    _esegui("src/assegnazioni.py")
    _esegui("src/prolog_engine.py")
    _esegui("src/bayes_engine.py")
    _esegui("src/train_models.py")
    if sensibilita:
        _esegui("src/sensitivita.py")
    _esegui("src/valutazione_dispatch.py")
    plotter.tutti()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", nargs="?", default="demo", choices=["demo", "esperimenti", "grafici"])
    parser.add_argument("--senza-sensibilita", action="store_true", help="salta src/sensitivita.py")
    args = parser.parse_args()
    if args.comando == "demo":
        demo()
    elif args.comando == "esperimenti":
        esperimenti(not args.senza_sensibilita)
    else:
        plotter.tutti()
