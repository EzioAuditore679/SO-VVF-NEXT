"""Parametri di esecuzione per tutti i moduli"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Valore del seme
SEME = 42

# Dataset
N_TESTI_SINTETICI = 1000
N_FOLD_CV = 5
RUMORE_GRAVITA = 0.12  # probabilità di un evento di rumore sull'etichetta (assunzione dichiarata);
# agli estremi metà degli eventi è assorbita, tasso effettivo circa 0.085 (relazione, sezione 1)
LAPLACE_ALPHA = 1.0  # pseudoconteggio per cella delle CPT della rete bayesiana

# Mappa e ricerca del percorso (src/ricerca.py)
DIM_GRIGLIA = (30, 30)
PROB_OSTACOLO = 0.2      # ostacoli permanenti (edifici)
PROB_CELLA_LENTA = 0.15  # celle lente (centro storico, traffico): entrarci costa COSTO_CELLA_LENTA
COSTO_CELLA_LENTA = 3
PROB_CHIUSURA = 0.05     # chiusure temporanee negli scenari di dispatch
PROB_CHIUSURE = [0.0, 0.05, 0.10, 0.20]  # livelli di chiusure nel confronto degli algoritmi
N_MAPPE = 30             # mappe per livello di chiusure, un incidente ciascuna
N_MEZZI_MAPPA = 8        # mezzi per incidente: la tabella della PD è riusata da tutti
LIMITE_NODI = 500_000    # oltre questo numero di nodi la ricerca è segnata come non completata

# Assegnazione dei mezzi: ottimizzazione vincolata con simulated annealing (src/assegnazioni.py)
N_INCIDENTI_SCENARI = [2, 4, 6, 8, 10]
# Un incidente richiede in media 1.6 ruoli (misurato sul modello generativo).
N_MEZZI_PER_INCIDENTE = 2.2  # mezzi disponibili = ceiling(incidenti * questo valore)
# Composizione della flotta, proporzionale alla domanda media di ciascun ruolo.
COMPOSIZIONE_FLOTTA = {"autopompa": 0.50, "nucleo_nbcr": 0.20, "canadair": 0.15, "autoscala": 0.15}
# Costo di un ruolo lasciato scoperto (rinforzi da un comando vicino). Supera il tempo di percorrenza
# tipico sulla mappa, quindi il dispatch lascia scoperto un ruolo solo se non ha alternative.
PENALITA_RUOLO_SCOPERTO = 120
DISTANZA_ADIACENZA = 3  # due incidenti a distanza di Manhattan <= 3 sono adiacenti
N_SCENARI_PER_DIMENSIONE = 20  # si riportano media ± deviazione standard
# Costo di una violazione del vincolo "un mezzo per ruolo": appena sopra la penalità, così un conflitto
# si risolve sempre lasciando scoperto uno dei due ruoli, ma resta attraversabile dalla ricerca locale.
COSTO_CONFLITTO = PENALITA_RUOLO_SCOPERTO + 1
# Vincolo flessibile: per ogni incidente critico si paga anche il tempo di arrivo dell'ultimo mezzo della
# squadra, pesato così. Con peso 1 un minuto di attesa della squadra completa vale un minuto di viaggio.
PESO_COMPLETAMENTO = 1.0
# Simulated annealing: valori scelti con la taratura in src/assegnazioni.py (scenari con seme SEME + 500, relazione sezione 4).
SA_PASSI = 50_000      # passi per tentativo
SA_T_INIZIALE = 100.0  # a inizio ricerca un peggioramento di 10 minuti si accetta con probabilità 0.9
SA_T_FINALE = 0.1      # a fine ricerca i peggioramenti non si accettano quasi più
SA_RIPARTENZE = 5      # tentativi indipendenti, si tiene il migliore

# Valutazione end-to-end delle fonti di gravità sul dispatch
N_INCIDENTI_E2E = 5
N_SCENARI_E2E = 100
SEME_E2E = SEME + 1000  # scenari indipendenti dal dataset di addestramento

# Percorsi
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
