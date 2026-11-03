"""Interfaccia Python verso la knowledge base Prolog (SWI-Prolog tramite pyswip).

Consulta knowledge_base/kb.pl, carica i fatti di uno scenario e interroga le regole.

Eseguito come script misura il costo del ragionamento in inferenze di SWI-Prolog (statistics(inferences, _)):
1. sugli scenari della valutazione end-to-end (stesso seme), inferenze per derivare squadra e mezzi adatti di
   tutti gli incidenti di uno scenario (results/kb_costo_scenari.csv);
2. caso peggiore della propagazione: gravita_operativa/2 di un incidente alto quando esiste un incidente critico
   non raggiungibile, con gli incidenti alti a catena oppure tutti adiacenti tra loro; esposto/1 fallisce solo
   dopo aver esplorato tutti i cammini semplici (results/kb_costo_propagazione.csv).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from domain import VALORE_IGNOTO
from pyswip import Prolog

KB_PATH = config.KB_DIR / "kb.pl"
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_DINAMICI = [
    "incidente(_,_,_,_,_,_)",
    "registro(_,_,_)",
    "adiacenza(_,_)",
    "mezzo(_,_)",
    "velocita_stimata(_,_)",
    "gravita_stimata(_,_)",
    "assegnato(_,_,_)",
]
RUOLI = ["estinzione", "rinforzo", "soccorso_quota", "decontaminazione"]


def _valore(x) -> str:
    """Atomo Prolog per un valore di campo. Un campo mancante (None) diventa l'atomo ignoto: senza
    questa conversione 'None' sarebbe letto da Prolog come una variabile, che unifica con tutto."""
    if x is None:
        return VALORE_IGNOTO
    return _id(x.value if hasattr(x, "value") else x)


def _id(x) -> str:
    s = str(x)
    if not _ID.match(s):
        raise ValueError(f"identificatore non valido per Prolog: {s!r}")
    return s


class KnowledgeBase:

    def __init__(self, percorso: Path = KB_PATH):
        self._p = Prolog()
        self._p.consult(str(percorso))

    def _q(self, goal: str) -> list[dict]:
        return list(self._p.query(goal))

    def reset(self) -> None:
        self._q(", ".join(f"retractall({d})" for d in _DINAMICI))

    """ Carica i fatti in uno scenario"""
    def carica_scenario(self, incidenti: list[dict], mezzi: dict, registro: dict,
                        adiacenze: list[tuple], velocita: dict | None = None,
                        gravita_stimata: dict | None = None) -> None:
        self.reset()
        for inc in incidenti:
            self._q(
                f"assertz(incidente({_id(inc['id'])}, {_valore(inc['tipo'])}, {int(inc['piano'])}, "
                f"{_valore(inc['sostanze'])}, {_valore(inc['persone'])}, {_valore(inc['meteo'])}))"
            )
        for m, tipo in mezzi.items():
            self._q(f"assertz(mezzo({_id(m)}, {_valore(tipo)}))")
        for i, (vuln, n_adj) in registro.items():
            self._q(f"assertz(registro({_id(i)}, {_valore(vuln)}, {int(n_adj)}))")
        for a, b in adiacenze:
            self._q(f"assertz(adiacenza({_id(a)}, {_id(b)})), assertz(adiacenza({_id(b)}, {_id(a)}))")
        for i, v in (velocita or {}).items():
            self._q(f"assertz(velocita_stimata({_id(i)}, {_valore(v)}))")
        for i, g in (gravita_stimata or {}).items():
            self._q(f"assertz(gravita_stimata({_id(i)}, {_valore(g)}))")

    def mezzi_idonei(self, incidente: str) -> list[str]:
        r = self._q(f"mezzi_idonei({_id(incidente)}, L)")
        return sorted(r[0]["L"]) if r else []

    def gravita(self, incidente: str) -> str | None:
        r = self._q(f"gravita({_id(incidente)}, G)")
        return str(r[0]["G"]) if r else None

    def gravita_operativa(self, incidente: str) -> str | None:
        r = self._q(f"gravita_operativa({_id(incidente)}, G)")
        return str(r[0]["G"]) if r else None

    def spiegazioni_velocita(self, incidente: str, gravita: str) -> list[str]:
        """Velocità del fuoco che compaiono in almeno una spiegazione minima della gravità osservata."""
        r = self._q(f"spiegazioni_velocita({_id(incidente)}, {_valore(gravita)}, L)")
        return [str(v) for v in r[0]["L"]] if r else []

    def spiegazioni(self, osservazioni: list[tuple[str, str]]) -> list[list[str]]:
        """Spiegazioni minime e coerenti di un insieme di osservazioni (incidente, gravità osservata).

        Ogni spiegazione è un insieme di assunzioni, ad esempio ['vel(c1,alta)']. Una lista vuota
        significa che le osservazioni sono incoerenti con la KB (nessun insieme di assunzioni le spiega)."""
        oss = ", ".join(f"gravita_osservata({_id(i)}, {_valore(g)})" for i, g in osservazioni)
        r = self._q(f"spiegazioni([{oss}], Hs), "
                    f"findall(S, (member(H, Hs), format(atom(S), '~w', [H])), L)")
        return [self._lista(str(s)) for s in r[0]["L"]] if r else []

    @staticmethod
    def _lista(testo: str) -> list[str]:
        """'[vel(c1,alta),pers(c1,incerte)]' -> ['vel(c1,alta)', 'pers(c1,incerte)']"""
        return re.findall(r"\w+\([^()]*\)", testo)

    def esposto(self, incidente: str) -> bool:
        return bool(self._q(f"esposto({_id(incidente)})"))

    def collegati(self, incidente: str) -> set[str]:
        """Incidenti raggiungibili per adiacenza, escluso l'incidente stesso."""
        r = self._q(f"setof(B, collegato({_id(incidente)}, B), L) -> true ; L = []")
        return (set(r[0]["L"]) if r else set()) - {incidente}

    def squadra(self, incidente: str) -> list[str]:
        """Ruoli richiesti dall'incidente, nell'ordine di RUOLI."""
        r = self._q(f"squadra_richiesta({_id(incidente)}, L)")
        ruoli = {str(x) for x in r[0]["L"]} if r else set()
        return [x for x in RUOLI if x in ruoli]

    def mezzi_adatti(self, incidente: str, ruolo: str) -> list[str]:
        r = self._q(f"mezzi_adatti({_id(incidente)}, {_id(ruolo)}, L)")
        return sorted(str(m) for m in r[0]["L"]) if r else []

    def motivi(self, incidente: str) -> list[tuple[str, str]]:
        """Spiegazione della squadra: coppie (ruolo, motivo) prodotte da motivo/3."""
        r = self._q(f"findall(S, (motivo({_id(incidente)}, R, M), format(atom(S), '~w|~w', [R, M])), L)")
        coppie = []
        for x in (r[0]["L"] if r else []):
            ruolo, motivo = str(x).split("|", 1)
            coppie.append((ruolo, motivo))
        return sorted(set(coppie), key=lambda c: (RUOLI.index(c[0]), c[1]))

    def _con_assegnazioni(self, assegnazioni: list[tuple], goal: str) -> list[str]:
        self._q("retractall(assegnato(_,_,_))")
        for m, i, ruolo in assegnazioni:
            self._q(f"assertz(assegnato({_id(m)}, {_id(i)}, {_id(ruolo)}))")
        r = self._q(goal)
        self._q("retractall(assegnato(_,_,_))")
        return sorted(str(x) for x in r[0]["L"]) if r else []

    def violazioni(self, assegnazioni: list[tuple]) -> list[str]:
        return self._con_assegnazioni(
            assegnazioni, "findall(S, (violazione(V), format(atom(S), '~w', [V])), L)")

    def conflitti(self, assegnazioni: list[tuple]) -> list[list[str]]:
        """Conflitti minimi della proposta: insiemi minimi di assegnazioni che implicano falso."""
        r = self._con_assegnazioni(
            assegnazioni, "conflitti_minimi(Cs), findall(S, (member(C, Cs), format(atom(S), '~w', [C])), L)")
        return [self._lista(c) for c in r]

    def ruoli_scoperti(self, assegnazioni: list[tuple]) -> list[str]:
        return self._con_assegnazioni(
            assegnazioni, "findall(S, (ruolo_scoperto(I, R), format(atom(S), '~w-~w', [I, R])), L)")

    def fatti_dominio(self) -> dict:
        def coppie(pred):
            return {str(r["K"]): int(r["V"]) for r in self._q(f"{pred}(K, V)")}

        def valore(pred):
            return int(self._q(f"{pred}(V)")[0]["V"])

        return {
            "punti_persone": coppie("punti_persone"),
            "punti_velocita": coppie("punti_velocita"),
            "soglia_critico": valore("soglia_critico"),
            "soglia_alto": valore("soglia_alto"),
            "soglia_piani_autoscala": valore("soglia_piani_autoscala"),
        }

    def inferenze(self, goal: str) -> int:
        """Numero di inferenze di SWI-Prolog per dimostrare goal (tutte le soluzioni, con findall)."""
        r = self._q(f"statistics(inferences, I0), findall(x, ({goal}), _), statistics(inferences, I1), "
                    f"N is I1 - I0")
        return int(r[0]["N"])

    def complessita(self) -> dict:
        r = self._q("kb_statistiche(P, R, C)")
        return {"predicati": int(r[0]["P"]), "regole": int(r[0]["R"]), "clausole": int(r[0]["C"])}


# --- Esperimento: costo del ragionamento ---

N_COLLEGATI = [3, 5, 7, 8, 9]


def costo_scenari_e2e(kb: KnowledgeBase):
    import math
    import random

    import pandas as pd

    import assegnazioni as asg  # import locali: servono solo a questo esperimento
    import generate_dataset as gd

    rng = random.Random(config.SEME_E2E)
    n_mezzi = math.ceil(config.N_INCIDENTI_E2E * config.N_MEZZI_PER_INCIDENTE)
    righe = []
    for k in range(config.N_SCENARI_E2E):
        caso = asg.genera_caso(rng, config.N_INCIDENTI_E2E, n_mezzi, variabili=lambda r: gd.genera_record(r, 0))
        asg.ruoli_dalla_kb(kb, caso)
        goal = ", ".join(f"squadra_richiesta({i}, _), forall(ruolo({i}, R), mezzi_adatti({i}, R, _))"
                         for i in sorted(caso.incidenti))
        righe.append(dict(scenario=k, adiacenze=len(asg.adiacenze(caso)), inferenze=kb.inferenze(goal)))
    return pd.DataFrame(righe)


def _incidente_prova(i, persone):
    return dict(id=i, tipo="industriale", piano=1, sostanze="no", persone=persone, meteo="calmo")


def costo_caso_peggiore(kb: KnowledgeBase):
    """Un incidente critico isolato e n incidenti alti collegati; si misura gravita_operativa del primo."""
    import pandas as pd

    righe = []
    for forma in ["a catena", "tutti adiacenti tra loro"]:
        for n in N_COLLEGATI:
            alti = [f"a{k}" for k in range(n)]
            incidenti = [_incidente_prova("crit", "confermate")] + [_incidente_prova(a, "confermate") for a in alti]
            registro = {"crit": ("alta", 2), **{a: ("bassa", 0) for a in alti}}
            velocita = {"crit": "alta", **{a: "media" for a in alti}}  # alti: 2 + 1 = 3 punti
            if forma == "a catena":
                adiacenze = list(zip(alti, alti[1:]))
            else:
                adiacenze = [(x, y) for k, x in enumerate(alti) for y in alti[k + 1:]]
            kb.carica_scenario(incidenti, {}, registro, adiacenze, velocita)
            assert kb.gravita("crit") == "critico" and kb.gravita_operativa("a0") == "alto"
            righe.append(dict(forma=forma, incidenti_alti=n, inferenze=kb.inferenze("gravita_operativa(a0, _)")))
    return pd.DataFrame(righe)


if __name__ == "__main__":
    kb = KnowledgeBase()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    e2e = costo_scenari_e2e(kb)
    e2e.to_csv(config.RESULTS_DIR / "kb_costo_scenari.csv", index=False)
    print(f"Squadra e mezzi adatti per scenario: {e2e.inferenze.mean():.0f} ± {e2e.inferenze.std():.0f} inferenze "
          f"(massimo {e2e.inferenze.max()}); adiacenze per scenario {e2e.adiacenze.mean():.2f}")
    peggiore = costo_caso_peggiore(kb)
    peggiore.to_csv(config.RESULTS_DIR / "kb_costo_propagazione.csv", index=False)
    print(peggiore.pivot(index="forma", columns="incidenti_alti", values="inferenze").to_string())
    print("Complessità della KB:", kb.complessita())
