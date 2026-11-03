"""Classificazione della gravità con nested cross-validation.

Esperimento A: i campi estratti dal testo dall'adattatore NLP.
Esperimento B: A, più i fatti del registro (vulnerabilità, edifici adiacenti) e la
probabilità di gravità critica della rete bayesiana, discretizzata in fasce a quantili.
Rete: la rete bayesiana usata come classificatore, con la stessa evidenza di B tranne gli adiacenti.
Controlli: C = B senza la rete; oracolo = C con la velocità vera del fuoco.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

import bayes_engine as be
import config
from domain import Meteo, Persone, SostanzePericolose, TipoEdificio, VelocitaFuoco, Vulnerabilita
from nlp_extract import estrai_campi

CAMPI_TESTO = ["tipo_edificio", "piano", "sostanze_pericolose", "persone", "meteo"]
CAMPI_REGISTRO = ["vulnerabilita", "adiacenti"]
N_FASCE_PROB = 5  # fasce a quantili di P(critico), calcolate sul fold di addestramento
NON_RICONOSCIUTO = "nessuno"  # campo che l'adattatore NLP non trova nel testo

# Domini delle feature categoriche, presi dal vocabolario del dominio e non dai dati: un valore raro che
# manca in un fold di addestramento resta comunque una categoria nota (con il solo pseudoconteggio).
CATEGORIE = {
    "tipo_edificio": [e.value for e in TipoEdificio] + [NON_RICONOSCIUTO],
    "piano": [str(p) for p in range(7)],
    "sostanze_pericolose": [e.value for e in SostanzePericolose],
    "persone": [e.value for e in Persone] + [NON_RICONOSCIUTO],
    "meteo": [e.value for e in Meteo] + [NON_RICONOSCIUTO],
    "vulnerabilita": [e.value for e in Vulnerabilita],
    "adiacenti": [str(n) for n in range(4)],
    "velocita_fuoco": [e.value for e in VelocitaFuoco],
}
# Codice dei valori fuori dominio (ad esempio un piano oltre il sesto letto nel testo).
CODICE_IGNOTO = max(len(v) for v in CATEGORIE.values())


def campi_estratti(testo: pd.Series) -> pd.DataFrame:
    ev = pd.DataFrame([estrai_campi(t) for t in testo])[CAMPI_TESTO]
    ev = ev.fillna(NON_RICONOSCIUTO)
    ev["piano"] = ev["piano"].astype(str)
    return ev


"""Per ogni modello: stimatore e griglia degli iperparametri"""
def modelli(seme: int) -> dict:

    return {
        "baseline_maggioranza": (DummyClassifier(strategy="most_frequent"), {}),
        "regressione_logistica": (LogisticRegression(max_iter=2000),
                                  {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "random_forest": (RandomForestClassifier(n_estimators=200, random_state=seme),
                          {"clf__max_depth": [3, 6, None]}),
        # Naive Bayes categorico: una distribuzione P(X_i | gravità) per feature, con pseudoconteggi alpha.
        "naive_bayes": (CategoricalNB(min_categories=CODICE_IGNOTO + 1),
                        {"clf__alpha": [0.01, 0.1, 1, 10, 100]}),
    }

"""Colonne categoriali e P(critico), se presente, discretizzata in fasce a quantili.

Regressione logistica e random forest ricevono la codifica one-hot; naive Bayes categorico riceve un codice
intero per valore (una variabile categorica per feature, come nel modello delle dispense)."""
def _pipeline(stimatore, colonne: list) -> Pipeline:
    categoriche = [c for c in colonne if c != "p_critico"]
    if isinstance(stimatore, CategoricalNB):
        codifica = OrdinalEncoder(categories=[CATEGORIE[c] for c in categoriche],
                                  handle_unknown="use_encoded_value", unknown_value=CODICE_IGNOTO)
        fasce = "ordinal"
    else:
        codifica = OneHotEncoder(handle_unknown="ignore")
        fasce = "onehot"
    parti = [("cat", codifica, categoriche)]
    if "p_critico" in colonne:
        parti.append(("prob", KBinsDiscretizer(N_FASCE_PROB, encode=fasce, strategy="quantile",
                                               quantile_method="averaged_inverted_cdf"), ["p_critico"]))
    return Pipeline([("pre", ColumnTransformer(parti)), ("clf", stimatore)])


METRICHE = ["accuracy", "f1_macro", "f1_pesata"]

"""Accuratezza, F1 macro (ogni classe pesa uguale) e F1 pesata (ogni classe pesa quanto la sua frequenza)."""
def metriche(y_vero, y_pred) -> dict:
    return dict(
        accuracy=accuracy_score(y_vero, y_pred),
        f1_macro=f1_score(y_vero, y_pred, average="macro"),
        f1_pesata=f1_score(y_vero, y_pred, average="weighted"),
    )

"""Media e deviazione standard di ogni metrica sui fold esterni."""
def riassumi(righe: list[dict], chiavi: list) -> pd.DataFrame:
    gruppi = pd.DataFrame(righe).groupby(chiavi)
    aggregati = {}
    for m in METRICHE:
        aggregati[f"{m}_media"] = (m, "mean")
        aggregati[f"{m}_dev"] = (m, "std")
    return gruppi.agg(**aggregati).reset_index()


def _fold_esterni(y: np.ndarray, seme: int):
    return StratifiedKFold(config.N_FOLD_CV, shuffle=True, random_state=seme).split(np.zeros(len(y)), y)


"""Nested CV: esterna per la stima, interna per gli iperparametri (scoring F1 macro).

probabilita: per ogni fold esterno, (p_tr, p_te) della rete bayesiana. Se presente,
diventa la feature 'p_critico', discretizzata dentro la pipeline.
"""
def _valuta(esperimento: str, X: pd.DataFrame, y: np.ndarray, probabilita: list | None, seme: int) -> list[dict]:
    righe = []
    for fold, (tr, te) in enumerate(_fold_esterni(y, seme)):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        if probabilita is not None:
            X_tr["p_critico"], X_te["p_critico"] = probabilita[fold]
        for nome, (stimatore, griglia) in modelli(seme).items():
            pipe = _pipeline(stimatore, list(X_tr.columns))
            if griglia:
                interna = StratifiedKFold(config.N_FOLD_CV, shuffle=True, random_state=seme)
                ricerca = GridSearchCV(pipe, griglia, cv=interna, scoring="f1_macro").fit(X_tr, y[tr])
                modello, parametri = ricerca.best_estimator_, ricerca.best_params_
            else:
                modello, parametri = pipe.fit(X_tr, y[tr]), {}
            righe.append(dict(esperimento=esperimento, modello=nome, fold=fold,
                              **metriche(y[te], modello.predict(X_te)), parametri=str(parametri)))
    return righe


"""Rete bayesiana su ogni fold esterno: P(critico) per l'esperimento B e la rete come classificatore."""
def _rete_per_fold(df: pd.DataFrame, evidenza: pd.DataFrame, y: np.ndarray, seme: int) -> tuple[list, list]:
    colonne = be.OSSERVATE_CON_REGISTRO
    probabilita, righe = [], []
    for fold, (tr, te) in enumerate(_fold_esterni(y, seme)):
        alpha = be.scegli_alpha(df.iloc[tr], evidenza.iloc[tr], colonne, seme=seme)
        p_tr = be.p_critico_fuori_fold(df.iloc[tr], evidenza.iloc[tr], seme=seme, colonne=colonne, alpha=alpha)
        rete = be.stima(df.iloc[tr], alpha)
        probabilita.append((p_tr, be.p_critico_righe(rete, evidenza.iloc[te], colonne)))
        pred = be.gravita_piu_probabile(rete, evidenza.iloc[te], colonne)
        righe.append(dict(esperimento="rete", modello="rete_bayesiana", fold=fold,
                          **metriche(y[te], pred), parametri=str({"alpha": alpha})))
    return probabilita, righe

"""Esperimenti A, B, rete, C e oracolo su un dataset. Una riga per modello e fold esterno."""
def esegui(df: pd.DataFrame, seme: int = config.SEME) -> list[dict]:
    y = df["gravita"].values
    X_a = campi_estratti(df["testo"])  # A: solo i campi del testo

    X_c = X_a.copy()  # B e C: testo + registro. B aggiunge P(critico) della rete, C ne isola il contributo
    for col in CAMPI_REGISTRO:
        X_c[col] = df[col].astype(str).values

    ev_rete = X_a.copy()  # la rete vede il testo e la vulnerabilità, non gli adiacenti
    ev_rete["vulnerabilita"] = df["vulnerabilita"].astype(str).values
    probabilita, righe_rete = _rete_per_fold(df, ev_rete, y, seme)

    X_o = X_c.copy()  # oracolo: C più la velocità vera del fuoco, non disponibile a runtime
    X_o["velocita_fuoco"] = df["velocita_fuoco"].astype(str).values

    return (_valuta("A", X_a, y, None, seme)
            + _valuta("B", X_c, y, probabilita, seme)
            + righe_rete
            + _valuta("C", X_c, y, None, seme)
            + _valuta("oracolo", X_o, y, None, seme))


def riepilogo(righe: list[dict]) -> pd.DataFrame:
    return riassumi(righe, ["esperimento", "modello"])


if __name__ == "__main__":
    df = pd.read_csv(config.DATA_DIR / "dataset_sintetico.csv")
    righe = esegui(df)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(righe).to_csv(config.RESULTS_DIR / "ml_esiti_fold.csv", index=False)
    tab = riepilogo(righe)
    tab.to_csv(config.RESULTS_DIR / "ml_riepilogo.csv", index=False)
    print(tab.round(3).to_string(index=False))
