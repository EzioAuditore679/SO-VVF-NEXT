"""Vocabolario unico del dominio SO-VVF-NEXT.

Generatore, adattatore NLP, KB Prolog, rete bayesiana, ricerca e ML importano da qui,
così le etichette non possono divergere tra i moduli.
"""

from enum import Enum


class TipoEdificio(str, Enum):
    RESIDENZIALE = "residenziale"
    INDUSTRIALE = "industriale"
    BOSCO = "bosco"


class SostanzePericolose(str, Enum):
    SI = "si"
    NO = "no"


"""Gravità a tre livelli. La regola che la definisce è in modello_generativo.regola_gravita e in kb.pl."""
class Gravita(str, Enum):
    CRITICO = "critico"
    ALTO = "alto"
    BASSO = "basso"


class VelocitaFuoco(str, Enum):
    ALTA = "alta"
    MEDIA = "media"
    BASSA = "bassa"


class Persone(str, Enum):
    CONFERMATE = "confermate"
    INCERTE = "incerte"
    ASSENTI = "assenti"


class Meteo(str, Enum):
    VENTO_FORTE = "vento_forte"
    CALMO = "calmo"


"""Vulnerabilità strutturale della zona: fatto del registro territoriale, non presente nella chiamata."""
class Vulnerabilita(str, Enum):
    ALTA = "alta"
    BASSA = "bassa"


class TipoMezzo(str, Enum):
    AUTOPOMPA = "autopompa"
    AUTOSCALA = "autoscala"
    NUCLEO_NBCR = "nucleo_nbcr"  # emergenze nucleari, biologiche, chimiche, radiologiche
    CANADAIR = "canadair"


class Campo(str, Enum):
    """Campi estratti dal testo della chiamata."""
    TIPO_EDIFICIO = "tipo_edificio"
    SOSTANZE_PERICOLOSE = "sostanze_pericolose"
    PERSONE = "persone"


"""Frasi usate dal generatore per i campi del testo. L'estrattore NLP ha un lessico suo"""
PAROLE_CHIAVE: dict[Campo, dict[Enum, list[str]]] = {
    Campo.TIPO_EDIFICIO: {
        TipoEdificio.RESIDENZIALE: ["una palazzina", "un condominio", "un appartamento"],
        TipoEdificio.INDUSTRIALE: ["un capannone", "uno stabilimento", "un deposito"],
        TipoEdificio.BOSCO: ["un bosco", "delle sterpaglie", "una pineta"],
    },
    Campo.SOSTANZE_PERICOLOSE: {
        SostanzePericolose.SI: ["bombole di gas", "un serbatoio", "delle vernici", "dei solventi", "dell'acido"],
        SostanzePericolose.NO: ["nessuna sostanza pericolosa", "niente sostanze pericolose", "nessun rischio chimico"],
    },
    Campo.PERSONE: {
        Persone.CONFERMATE: ["ci sono persone intrappolate", "c'è qualcuno bloccato", "qualcuno non riesce a uscire"],
        Persone.INCERTE: ["non sappiamo se ci sia qualcuno", "forse c'è qualcuno dentro", "potrebbe esserci qualcuno"],
        Persone.ASSENTI: ["non c'è nessuno dentro", "tutti sono usciti", "l'edificio è vuoto"],
    },
}

# Altezza in piani oltre la quale serve l'autoscala. Assunzione dichiarata.
SOGLIA_PIANI_AUTOSCALA = 4

# Valore di un campo che l'adattatore NLP non riconosce nel testo. La KB lo tratta in modo prudente
# (persone ignote come confermate, sostanze ignote come presenti); la rete bayesiana lo esclude
# dall'evidenza, cioè marginalizza la variabile.
VALORE_IGNOTO = "ignoto"
