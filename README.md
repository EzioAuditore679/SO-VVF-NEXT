# SO-VVF-NEXT

SO-VVF-NEXT è un prototipo di sistema basato su conoscenza per il dispatch dei mezzi in una sala operativa dei Vigili del Fuoco.

Il sistema legge il testo delle chiamate al 115, ne ricava i campi, stima la gravità di ogni incidente, deriva dalla Knowledge Base Prolog la squadra richiesta, assegna i mezzi ai ruoli (dispatch) con il simulated annealing e calcola il percorso di ogni mezzo con A*. Infine cerca in Prolog i conflitti minimi della proposta (diagnosi basata sulla consistenza).

Progetto per l'esame di Ingegneria della Conoscenza (Cdl Informatica UniBa)

---

## Obiettivo del progetto

L'obiettivo è realizzare un Knowledge-Based System che integri rappresentazione e ragionamento, ricerca, vincoli, ragionamento probabilistico e apprendimento in un'unica catena decisionale.

Data una chiamata come:

```text
Salve, abbiamo un incendio in un capannone al primo piano. Vicino alle fiamme ci sono bombole di gas.
Mi dicono che c'è qualcuno bloccato. Il vento è forte.
```

insieme alle altre chiamate aperte e alla flotta disponibile, il sistema produce:

- i campi della chiamata (tipo di edificio, piano, sostanze, persone, meteo);
- la gravità stimata da più fonti (ML, rete bayesiana, rete + regola della KB) e le spiegazioni abduttive minime delle stime;
- la gravità operativa, che tiene conto della propagazione tra incidenti vicini;
- la squadra richiesta (estinzione, rinforzo, soccorso in quota, decontaminazione), con il motivo di ogni ruolo;
- l'assegnazione dei mezzi ai ruoli, con i ruoli lasciati scoperti segnalati come richiesta di rinforzi;
- il percorso di ogni mezzo sulla mappa;
- i conflitti minimi della proposta (nessuno se è valida).

---

## Temi ricoperti

| Tema | Implementazione |
|---|---|
| Rappresentazione della conoscenza | Programma logico Prolog unico (`knowledge_base/kb.pl`): clausole definite estese con la negazione per fallimento |
| Ragionamento logico | Ricorsione con lista dei visitati, negazione per fallimento su goal ground, abduzione con assumibili e spiegazioni minime e coerenti (`dimostra/3`, `spiegazioni/2`), spiegazioni dei ruoli (`motivo/3`), conflitti minimi della proposta (`conflitti_minimi/1`) |
| Ricerca nello spazio degli stati | BFS, LcFS (A* con h = 0) e A* con potatura dei cammini multipli; euristica Manhattan e DB di pattern costruito offline con la programmazione dinamica (`DBPattern`), entrambe consistenti |
| Vincoli e ricerca locale | Assegnazione mezzi→ruoli come ottimizzazione vincolata, con un vincolo flessibile non separabile (completamento delle squadre critiche); simulated annealing con ripartenze, primo tentativo dalla soluzione dell'ungherese; ottimo esatto con programmazione lineare intera (HiGHS) |
| Ragionamento probabilistico | Rete bayesiana a 8 nodi con la velocità del fuoco nascosta a runtime (registrata nei rapporti d'intervento storici), CPT con pseudoconteggi (prior di Dirichlet), campi non riconosciuti marginalizzati |
| Apprendimento supervisionato | Classificazione della gravità: regressione logistica, random forest, naive Bayes categorico |
| Valutazione | Nested cross-validation 5×5, medie ± dev. std., analisi di sensibilità al rumore |
| Integrazione | Valutazione end-to-end delle fonti di gravità sul dispatch, su 100 scenari |

---

## Architettura del sistema

```text
Testo della chiamata
    |
    v
Adattatore NLP (regole)       -> tipo, piano, sostanze, persone, meteo
    |
    +--> Registro della zona   -> vulnerabilità, edifici adiacenti
    |
    v
Stima della gravità           -> ML  |  rete bayesiana  |  rete (velocità) + regola della KB
    |
    v
KB Prolog                     -> gravità operativa (propagazione), squadra, mezzi adatti, motivi
    |
    v
Simulated annealing           -> assegnazione mezzi→ruoli (con completamento delle squadre critiche)
    |
    v
Ricerca A* con h_PD           -> percorso di ogni mezzo (DB di pattern costruito offline)
    |
    v
KB Prolog                     -> conflitti minimi della proposta
```

---

## Struttura della repository

```text
so-vvf-next/
├── README.md
├── requirements.txt
├── pytest.ini
├── config.py                    # tutti i parametri: seme, dimensioni, rumore, mappa, simulated annealing
├── data/
│   └── dataset_sintetico.csv    # 1000 chiamate, rigenerabile
├── knowledge_base/
│   └── kb.pl                    # conoscenza di dominio e regole
├── docs/
│   └── Relazione_SO-VVF-NEXT.docx
├── results/                     # risultati in CSV
├── plots/                       # grafici
├── src/
│   ├── domain.py
│   ├── modello_generativo.py
│   ├── generate_dataset.py
│   ├── nlp_extract.py
│   ├── prolog_engine.py
│   ├── bayes_engine.py
│   ├── ricerca.py
│   ├── assegnazioni.py
│   ├── train_models.py
│   ├── sensitivita.py
│   ├── valutazione_dispatch.py
│   ├── plotter.py
│   └── main.py
└── tests/
    ├── test_prolog.py
    ├── test_ricerca.py
    ├── test_assegnazioni.py
    ├── test_bayes.py
    ├── test_nlp.py
    └── test_valutazione.py
```

---

## Installazione

Richiede Python 3.10+ e SWI-Prolog installato nel sistema (su Ubuntu/Debian `sudo apt-get install swi-prolog`; su Windows l'installer ufficiale, con `swipl` nel PATH).

```bash
python -m venv .venv
source .venv/bin/activate        # su Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dipendenze principali

- `numpy`
- `pandas`
- `scikit-learn`
- `pgmpy`
- `pyswip`
- `scipy`
- `matplotlib`
- `pytest`

---

## Demo e comandi principali

Eseguire la demo integrata (4 chiamate, pochi secondi):

```bash
python src/main.py
```

Rigenerare dataset, test, risultati e grafici (circa 17 minuti per l'esecuzione completa, circa 12 senza l'analisi di sensibilità):

```bash
python src/main.py esperimenti
python src/main.py esperimenti --senza-sensibilita
```

Rigenerare solo i grafici dai CSV:

```bash
python src/main.py grafici
```

Eseguire i singoli passi:

```bash
python src/generate_dataset.py       # dataset sintetico
python src/ricerca.py                # confronto BFS, LcFS e A* sul percorso, costo del DB offline
python src/assegnazioni.py           # taratura del simulated annealing e confronto con l'ottimo esatto
python src/prolog_engine.py          # costo del ragionamento della KB (inferenze) e complessità
python src/bayes_engine.py           # accuratezza della stima della velocità del fuoco
python src/train_models.py           # nested CV
python src/sensitivita.py            # sensibilità al rumore delle etichette
python src/valutazione_dispatch.py   # fonti di gravità valutate sul dispatch
```

Eseguire i test:

```bash
python -m pytest
```

---

## Output generati

```text
data/dataset_sintetico.csv
results/ricerca_percorsi.csv
results/ricerca_riepilogo.csv
results/ricerca_db_offline.csv
results/sa_taratura.csv
results/sa_costo_conflitti.csv
results/assegnazioni_scenari.csv
results/assegnazioni_riepilogo.csv
results/kb_costo_scenari.csv
results/kb_costo_propagazione.csv
results/stima_velocita.csv
results/ml_esiti_fold.csv
results/ml_riepilogo.csv
results/sensitivita_rumore.csv
results/dispatch_e2e_scenari.csv
results/dispatch_e2e_riepilogo.csv
plots/ml_modelli.png
plots/ricerca_percorsi.png
plots/dispatch_fonti.png
```

Tutti gli output sono rigenerabili con il seme fissato in `config.py`.

---

## Valutazione

Tutti i risultati sono riportati come media ± deviazione standard su più run.

**Classificazione della gravità** (nested CV 5×5, F1 macro, regressione logistica):

| Esperimento | Feature | F1 macro |
|---|---|---|
| A | campi del testo | 0.67 ± 0.04 |
| B | A + registro + P(critico) della rete | 0.72 ± 0.03 |
| C | B senza la rete (controllo) | 0.71 ± 0.04 |
| rete | rete bayesiana come classificatore | 0.70 ± 0.03 |
| oracolo | C + velocità vera del fuoco (controllo) | 0.91 ± 0.03 |

Il collo di bottiglia è la velocità del fuoco, nascosta al momento della chiamata (la rete la stima correttamente nel 54% dei casi, contro il 35% della classe più frequente): con la velocità vera l'accuratezza raggiunge il tetto imposto dal rumore delle etichette. B e C restano entro una deviazione standard: quasi tutto il guadagno viene dal registro. Le conclusioni restano le stesse con rumore 0.05, 0.12 e 0.20.

**Ricerca del percorso** (mappe 30×30, 240 percorsi per livello di chiusure): senza chiusure A* con h_PD espande in media 36 nodi, contro 92 di A* Manhattan e 335 di LcFS; il vantaggio cala con le chiusure. La BFS trova il percorso più veloce solo nel 28–46% dei casi, con un tempo in più del 6–14%. Il DB di pattern dipende solo da ostacoli e celle lente: si costruisce offline una volta per mappa (circa 700 tabelle da circa 715 nodi ciascuna, 1.3–1.8 s) e online A* legge la tabella dell'incidente.

**Assegnazione dei mezzi (dispatch)**: con il completamento delle squadre critiche la soluzione dell'ungherese non è più ottima (65–85% di ottimi con 4–10 incidenti). Con 5 tentativi, il primo dalla soluzione dell'ungherese, il simulated annealing trova sempre l'ottimo fino a 4 incidenti, nel 95% dei casi con 6 e nel 75% con 8–10, con uno scarto medio dello 0.37–0.56% (contro 0.73–0.84% dell'ungherese).

**Valutazione end-to-end** (100 scenari da 5 incidenti, giudicati con la gravità vera): su 76 incidenti critici, rete + KB ne manca 12, il classificatore ML 27 e la rete bayesiana 26; la fonte prudente non ne manca nessuno ma sovrastima molto. Le differenze di costo tra le fonti non sono significative.

Il progetto include 173 test automatici con `pytest` (più un caso atteso come fallito, documentato).
