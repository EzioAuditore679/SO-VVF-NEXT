"""Rete bayesiana con 8 nodi, CPT stimate con pseudoconteggi (prior di Dirichlet simmetrico; Laplace per alpha = 1)

La struttura è il grafo causale di modello_generativo.py, senza il numero di edifici adiacenti
(non modellato). La velocità del fuoco è nei dati di addestramento ma non è mai osservata in
inferenza. La vulnerabilità è osservata solo quando è noto il registro
della zona (OSSERVATE_CON_REGISTRO: esperimento B, rete come classificatore, dispatch).

Eseguito come script misura l'accuratezza della stima della velocità del fuoco (results/stima_velocita.csv).
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*pgmpy\..*")

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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

STATI = {
    "tipo_edificio": [e.value for e in TipoEdificio],
    "piano": [str(p) for p in range(7)],
    "sostanze_pericolose": [e.value for e in SostanzePericolose],
    "meteo": [e.value for e in Meteo],
    "velocita_fuoco": [e.value for e in VelocitaFuoco],
    "persone": [e.value for e in Persone],
    "vulnerabilita": [e.value for e in Vulnerabilita],
    "gravita": [e.value for e in Gravita],
}

ARCHI = [
    ("tipo_edificio", "piano"),
    ("tipo_edificio", "sostanze_pericolose"),
    ("tipo_edificio", "persone"),
    ("piano", "persone"),
    ("sostanze_pericolose", "velocita_fuoco"),
    ("meteo", "velocita_fuoco"),
    ("persone", "gravita"),
    ("velocita_fuoco", "gravita"),
    ("sostanze_pericolose", "gravita"),
    ("vulnerabilita", "gravita"),
]

# Variabili osservabili dal testo della chiamata: sono l'evidenza dell'inferenza.
OSSERVATE = ["tipo_edificio", "piano", "sostanze_pericolose", "meteo", "persone"]

# Con il registro della zona è nota anche la vulnerabilità (esperimento B, dispatch).
OSSERVATE_CON_REGISTRO = OSSERVATE + ["vulnerabilita"]
ALPHA_CANDIDATI = [0.1, 1.0, 10.0]  # griglia di alpha di Laplace per la CV interna


"""Converte le colonne in stringhe, per pgmpy."""
def prepara(df: pd.DataFrame) -> pd.DataFrame:
    
    return pd.DataFrame({nodo: df[nodo].astype(str) for nodo in STATI})


"""Stima le CPT dai conteggi, con alpha pseudoconteggi per cella (Laplace)."""
def stima(df: pd.DataFrame, alpha: float = config.LAPLACE_ALPHA) -> DiscreteBayesianNetwork:
    dati = prepara(df)
    modello = DiscreteBayesianNetwork(ARCHI)
    stimatore = BayesianEstimator(modello, dati, state_names=STATI)
    modello.add_cpds(*[
        stimatore.estimate_cpd(nodo, prior_type="dirichlet", pseudo_counts=alpha)
        for nodo in STATI
    ])
    if not modello.check_model():
        raise ValueError("CPT non valide")
    return modello


"""P(nodo | colonne) per ogni riga, una colonna per stato di STATI[nodo].

Le righe con la stessa evidenza condividono la query: le combinazioni distinte sono poche.
"""
def _query(modello: DiscreteBayesianNetwork, df: pd.DataFrame, colonne: list, nodo: str) -> np.ndarray:

    inferenza = VariableElimination(modello)
    dati = df[colonne].astype(str)
    risposte = {}
    out = np.empty((len(dati), len(STATI[nodo])))
    for k, riga in enumerate(dati.itertuples(index=False, name=None)):
        if riga not in risposte:
            # Un campo non riconosciuto dal testo (valore fuori dal dominio, es. "nessuno") non è
            # evidenza: la variabile resta non osservata e l'inferenza la marginalizza.
            evidenza = {c: v for c, v in zip(colonne, riga) if v in STATI[c]}
            q = inferenza.query([nodo], evidence=evidenza, show_progress=False)
            ordine = [q.state_names[nodo].index(s) for s in STATI[nodo]]
            risposte[riga] = q.values[ordine]
        out[k] = risposte[riga]
    return out

"""P(gravita | colonne) per ogni riga, colonne nell'ordine di STATI['gravita']."""
def distribuzione_gravita(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                          colonne: list = OSSERVATE) -> np.ndarray:
    return _query(modello, df, colonne, "gravita")


"""P(gravita = critico | colonne) per ogni riga."""
def p_critico_righe(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                    colonne: list = OSSERVATE) -> np.ndarray:
    return distribuzione_gravita(modello, df, colonne)[:, STATI["gravita"].index("critico")]


"""La rete come classificatore: lo stato di gravità con probabilità a posteriori massima."""
def gravita_piu_probabile(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                          colonne: list = OSSERVATE) -> np.ndarray:
    p = distribuzione_gravita(modello, df, colonne)
    return np.array(STATI["gravita"])[p.argmax(axis=1)]


"""Stima della variabile latente: la velocità del fuoco più probabile data l'evidenza."""
def velocita_piu_probabile(modello: DiscreteBayesianNetwork, df: pd.DataFrame,
                           colonne: list = OSSERVATE) -> np.ndarray:
    p = _query(modello, df, colonne, "velocita_fuoco")
    return np.array(STATI["velocita_fuoco"])[p.argmax(axis=1)]


"""Alpha di Laplace scelto in CV sui soli dati passati.
A parità di punteggio si preferisce l'alpha più piccolo.
"""
def scegli_alpha(df_stima: pd.DataFrame, df_evidenza: pd.DataFrame, colonne: list,
                 k: int = config.N_FOLD_CV, seme: int = config.SEME,
                 candidati: list = ALPHA_CANDIDATI) -> float:
    y = df_stima["gravita"].astype(str).values
    punteggi = {}
    for alpha in candidati:
        f1 = []
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seme).split(df_stima, y):
            modello = stima(df_stima.iloc[tr], alpha)
            pred = gravita_piu_probabile(modello, df_evidenza.iloc[te], colonne)
            f1.append(f1_score(y[te], pred, average="macro"))
        punteggi[alpha] = float(np.mean(f1))
    return max(candidati, key=lambda a: (punteggi[a], -a))


"""P(critico) per ogni riga di addestramento, senza usare la propria etichetta.

Ogni parte è prevista da una rete stimata sulle altre k-1 parti. df_stima contiene i valori
veri (per stimare le CPT); df_evidenza contiene le osservazioni che il sistema vede a runtime
(per l'inferenza). Serve come feature per il ML sui dati di addestramento.
"""
def p_critico_fuori_fold(df_stima: pd.DataFrame, df_evidenza: pd.DataFrame,
                         k: int = config.N_FOLD_CV, seme: int = config.SEME,
                         colonne: list = OSSERVATE, alpha: float = config.LAPLACE_ALPHA) -> np.ndarray:
    out = np.empty(len(df_stima))
    y = df_stima["gravita"].values
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seme).split(df_stima, y):
        modello = stima(df_stima.iloc[tr], alpha)
        out[te] = p_critico_righe(modello, df_evidenza.iloc[te], colonne)
    return out


"""P(nodo = stato | genitori) letta dalla CPT appresa."""
def probabilita(modello: DiscreteBayesianNetwork, nodo: str, stato, **genitori) -> float:
    cpd = modello.get_cpds(nodo)
    valori = {nodo: str(stato), **{k: str(v) for k, v in genitori.items()}}
    indice = tuple(cpd.state_names[v].index(valori[v]) for v in cpd.variables)
    return float(cpd.values[indice])


# --- Esperimento: accuratezza della stima della velocità del fuoco ---

"""Cross-validation stratificata a 5 fold: la rete è stimata sui fold di addestramento (dove la velocità
è registrata, come nei rapporti d'intervento) e predice la velocità più probabile sui fold di test usando
solo l'evidenza del testo, come a runtime. Riferimento: la classe più frequente del fold di addestramento."""
def valuta_stima_velocita(df: pd.DataFrame, seme: int = config.SEME) -> pd.DataFrame:

    from train_models import campi_estratti  

    evidenza = campi_estratti(df["testo"])
    y = df["velocita_fuoco"].astype(str).values
    righe = []
    for fold, (tr, te) in enumerate(StratifiedKFold(config.N_FOLD_CV, shuffle=True, random_state=seme).split(df, y)):
        rete = stima(df.iloc[tr])
        pred = velocita_piu_probabile(rete, evidenza.iloc[te], OSSERVATE)
        maggioranza = pd.Series(y[tr]).mode()[0]
        righe.append(dict(fold=fold, rete=float(np.mean(pred == y[te])),
                          classe_piu_frequente=float(np.mean(y[te] == maggioranza))))
    return pd.DataFrame(righe)


if __name__ == "__main__":
    tab = valuta_stima_velocita(pd.read_csv(config.DATA_DIR / "dataset_sintetico.csv"))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(config.RESULTS_DIR / "stima_velocita.csv", index=False)
    print("Stima della velocità del fuoco (accuratezza, CV a 5 fold):")
    for c in ["rete", "classe_piu_frequente"]:
        print(f"  {c}: {100 * tab[c].mean():.1f}% ± {100 * tab[c].std():.1f}%")
