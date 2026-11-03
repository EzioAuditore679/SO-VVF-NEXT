"""Test della knowledge base Prolog. Richiedono SWI-Prolog e pyswip, altrimenti vengono saltati."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
import modello_generativo as mg
from domain import (
    SOGLIA_PIANI_AUTOSCALA,
    Gravita,
    Meteo,
    Persone,
    SostanzePericolose,
    TipoEdificio,
    TipoMezzo,
    VelocitaFuoco,
    Vulnerabilita,
)


@pytest.fixture(scope="module")
def kb():
    pytest.importorskip("pyswip")
    from prolog_engine import KnowledgeBase
    try:
        return KnowledgeBase()
    except Exception as e:
        pytest.skip(f"SWI-Prolog non disponibile: {e}")


def _incidente(i, tipo=TipoEdificio.RESIDENZIALE, piano=0, sost=SostanzePericolose.NO,
               persone=Persone.ASSENTI, meteo=Meteo.CALMO):
    return dict(id=i, tipo=tipo, piano=piano, sostanze=sost, persone=persone, meteo=meteo)


def test_fatti_di_dominio_coincidono_con_python(kb):
    """Soglie e punteggi sono scritti a mano in kb.pl: devono restare uguali a quelli del generatore."""
    assert kb.fatti_dominio() == {
        "punti_persone": {p.value: n for p, n in mg.PUNTI_PERSONE.items()},
        "punti_velocita": {v.value: n for v, n in mg.PUNTI_VELOCITA.items()},
        "soglia_critico": mg.SOGLIA_CRITICO,
        "soglia_alto": mg.SOGLIA_ALTO,
        "soglia_piani_autoscala": SOGLIA_PIANI_AUTOSCALA,
    }


def test_autoscala_solo_in_altezza_e_con_persone(kb):
    conf = Persone.CONFERMATE
    inc = [_incidente("a3", TipoEdificio.RESIDENZIALE, 3, persone=conf),
           _incidente("a4", TipoEdificio.RESIDENZIALE, 4, persone=conf),
           _incidente("b5", TipoEdificio.INDUSTRIALE, 5, persone=conf),
           _incidente("v5", TipoEdificio.RESIDENZIALE, 5, persone=Persone.ASSENTI)]
    mezzi = {"m_scala": TipoMezzo.AUTOSCALA, "m_pompa": TipoMezzo.AUTOPOMPA}
    kb.carica_scenario(inc, mezzi, {i["id"]: (Vulnerabilita.BASSA, 0) for i in inc}, [],
                       {i["id"]: VelocitaFuoco.BASSA for i in inc})
    assert "m_scala" not in kb.mezzi_idonei("a3")
    assert "m_scala" in kb.mezzi_idonei("a4")
    assert "m_scala" in kb.mezzi_idonei("b5")
    assert "m_scala" not in kb.mezzi_idonei("v5")  # edificio alto ma nessuno da soccorrere


def test_negazione_canadair_non_con_sostanze(kb):
    inc = [_incidente("bosco1", TipoEdificio.BOSCO, 0, SostanzePericolose.NO),
           _incidente("bosco2", TipoEdificio.BOSCO, 0, SostanzePericolose.SI)]
    kb.carica_scenario(inc, {"cnd": TipoMezzo.CANADAIR}, {i["id"]: (Vulnerabilita.BASSA, 0) for i in inc},
                       [], {i["id"]: VelocitaFuoco.BASSA for i in inc})
    assert kb.mezzi_idonei("bosco1") == ["cnd"]
    assert kb.mezzi_idonei("bosco2") == []


def test_gravita_coerente_con_regola_python(kb):
    """Prolog e regola_gravita devono dare sempre la stessa gravità."""
    rng = random.Random(config.SEME)
    casi = [mg.genera_variabili(rng) for _ in range(2000)]
    inc, reg, vel, attesi = [], {}, {}, {}
    for n, v in enumerate(casi):
        i = f"c{n}"
        inc.append(_incidente(i, v["tipo_edificio"], v["piano"], v["sostanze_pericolose"],
                              v["persone"], v["meteo"]))
        reg[i] = (v["vulnerabilita"], v["adiacenti"])
        vel[i] = v["velocita_fuoco"]
        attesi[i] = mg.regola_gravita(v).value
    kb.carica_scenario(inc, {}, reg, [], vel)
    difformi = [i for i in attesi if kb.gravita(i) != attesi[i]]
    assert not difformi, f"gravità diverse da regola_gravita su {len(difformi)} casi, es. {difformi[:3]}"


def test_abduzione_contiene_la_velocita_vera(kb):
    rng = random.Random(config.SEME + 1)
    casi = [mg.genera_variabili(rng) for _ in range(300)]
    inc, reg, vel = [], {}, {}
    for n, v in enumerate(casi):
        i = f"h{n}"
        inc.append(_incidente(i, v["tipo_edificio"], v["piano"], v["sostanze_pericolose"],
                              v["persone"], v["meteo"]))
        reg[i] = (v["vulnerabilita"], v["adiacenti"])
        vel[i] = v["velocita_fuoco"]
    kb.carica_scenario(inc, {}, reg, [], vel)
    for n, v in enumerate(casi):
        i = f"h{n}"
        assert v["velocita_fuoco"].value in kb.spiegazioni_velocita(i, kb.gravita(i))


def test_propagazione_ricorsiva_e_ciclo_terminano(kb):
    # ca critica (punteggio 6); cb e cc, bassa, sono raggiungibili da ca anche attraverso il ciclo.
    inc = [_incidente("ca", TipoEdificio.INDUSTRIALE, 1, SostanzePericolose.NO, Persone.CONFERMATE),
           _incidente("cb", TipoEdificio.INDUSTRIALE, 1, SostanzePericolose.NO, Persone.ASSENTI),
           _incidente("cc", TipoEdificio.INDUSTRIALE, 1, SostanzePericolose.NO, Persone.ASSENTI)]
    reg = {"ca": (Vulnerabilita.ALTA, 2), "cb": (Vulnerabilita.BASSA, 0), "cc": (Vulnerabilita.BASSA, 0)}
    vel = {"ca": VelocitaFuoco.ALTA, "cb": VelocitaFuoco.BASSA, "cc": VelocitaFuoco.BASSA}
    kb.carica_scenario(inc, {}, reg, [("ca", "cb"), ("cb", "cc"), ("cc", "ca")], vel)
    assert kb.gravita("ca") == "critico"
    assert kb.collegati("ca") == {"cb", "cc"}
    assert kb.esposto("cb") is True
    assert kb.esposto("cc") is True
    assert kb.esposto("ca") is False


def test_gravita_operativa_eleva_alto_esposto_a_critico(kb):
    # i1 critica; i2 alta ma adiacente a i1.
    inc = [_incidente("i1", TipoEdificio.INDUSTRIALE, 2, SostanzePericolose.SI, Persone.CONFERMATE, Meteo.VENTO_FORTE),
           _incidente("i2", TipoEdificio.INDUSTRIALE, 0, SostanzePericolose.SI, Persone.ASSENTI, Meteo.VENTO_FORTE)]
    reg = {"i1": (Vulnerabilita.ALTA, 2), "i2": (Vulnerabilita.BASSA, 0)}
    vel = {"i1": VelocitaFuoco.ALTA, "i2": VelocitaFuoco.ALTA}
    kb.carica_scenario(inc, {}, reg, [("i1", "i2")], vel)
    assert kb.gravita("i2") == "alto"
    assert kb.gravita_operativa("i2") == "critico"
    assert kb.gravita("i1") == "critico"


def test_violazioni_proposta_valida_e_non(kb):
    inc = [_incidente("v1", TipoEdificio.RESIDENZIALE, 4, persone=Persone.CONFERMATE),
           _incidente("v2", TipoEdificio.BOSCO, 0)]
    mezzi = {"m_scala": TipoMezzo.AUTOSCALA, "m_pompa": TipoMezzo.AUTOPOMPA, "m_cnd": TipoMezzo.CANADAIR}
    kb.carica_scenario(inc, mezzi, {i["id"]: (Vulnerabilita.BASSA, 0) for i in inc}, [],
                       {i["id"]: VelocitaFuoco.BASSA for i in inc})
    valida = [("m_scala", "v1", "soccorso_quota"), ("m_pompa", "v1", "estinzione"), ("m_cnd", "v2", "estinzione")]
    assert kb.violazioni(valida) == []
    assert kb.ruoli_scoperti(valida) == []
    viol = kb.violazioni([("m_scala", "v1", "soccorso_quota"), ("m_scala", "v2", "estinzione"),
                          ("m_cnd", "v1", "estinzione")])
    assert any(v.startswith("doppia_assegnazione(m_scala)") for v in viol)
    assert any(v.startswith("non_adatto(m_cnd,v1,estinzione)") for v in viol)
    assert kb.ruoli_scoperti([("m_scala", "v1", "soccorso_quota")]) == ["v1-estinzione", "v2-estinzione"]


def _gravi(kb, gravita: dict, **campi):
    """Un incidente per voce di gravita, con la gravità imposta come stima esterna."""
    inc = [_incidente(i, **campi) for i in gravita]
    kb.carica_scenario(inc, {}, {i: (Vulnerabilita.BASSA, 0) for i in gravita}, [],
                       gravita_stimata=gravita)


def test_squadra_dipende_dalla_gravita(kb):
    _gravi(kb, {"g_c": Gravita.CRITICO, "g_a": Gravita.ALTO, "g_b": Gravita.BASSO},
           tipo=TipoEdificio.RESIDENZIALE, piano=5, persone=Persone.INCERTE)
    assert kb.squadra("g_c") == ["estinzione", "rinforzo", "soccorso_quota"]
    assert kb.squadra("g_a") == ["estinzione", "soccorso_quota"]
    assert kb.squadra("g_b") == ["estinzione"]  # persone incerte e gravità bassa: niente autoscala


def test_gravita_stimata_prevale_sulla_regola(kb):
    inc = [_incidente("s1", TipoEdificio.INDUSTRIALE, 1, SostanzePericolose.SI, Persone.CONFERMATE)]
    reg = {"s1": (Vulnerabilita.ALTA, 2)}
    kb.carica_scenario(inc, {}, reg, [], {"s1": VelocitaFuoco.ALTA}, {"s1": Gravita.BASSO})
    assert kb.gravita("s1") == "critico"            # regola con velocità
    assert kb.gravita_operativa("s1") == "basso"    # la stima esterna prevale
    assert kb.squadra("s1") == ["estinzione", "decontaminazione"]


def test_rinforzo_per_propagazione_con_spiegazione(kb):
    kb.carica_scenario([_incidente("p1"), _incidente("p2")], {},
                       {"p1": (Vulnerabilita.BASSA, 0), "p2": (Vulnerabilita.BASSA, 0)},
                       [("p1", "p2")], gravita_stimata={"p1": Gravita.CRITICO, "p2": Gravita.ALTO})
    assert "rinforzo" in kb.squadra("p2")
    assert ("rinforzo", "esposto_a(p1)") in kb.motivi("p2")
    assert ("rinforzo", "gravita_critica") in kb.motivi("p1")


def test_ogni_ruolo_ha_un_motivo(kb):
    rng = random.Random(config.SEME + 2)
    casi = [mg.genera_variabili(rng) for _ in range(300)]
    inc, reg, vel = [], {}, {}
    for n, v in enumerate(casi):
        i = f"r{n}"
        inc.append(_incidente(i, v["tipo_edificio"], v["piano"], v["sostanze_pericolose"],
                              v["persone"], v["meteo"]))
        reg[i] = (v["vulnerabilita"], v["adiacenti"])
        vel[i] = v["velocita_fuoco"]
    kb.carica_scenario(inc, {}, reg, [], vel)
    for n in range(len(casi)):
        i = f"r{n}"
        assert set(kb.squadra(i)) == {r for r, _ in kb.motivi(i)}


def test_complessita_kb_riportata(kb):
    c = kb.complessita()
    assert c["predicati"] > 0 and c["regole"] > 0 and c["clausole"] >= c["regole"]


@pytest.mark.parametrize("ordine", [["p1", "p2"], ["p2", "p1"]])
def test_propagazione_non_dipende_dall_ordine_dei_fatti(kb, ordine):
    """gravita_base/2 chiamata con l'incidente libero (in esposto/1) deve considerare tutti gli incidenti."""
    stime = {"p1": Gravita.CRITICO, "p2": Gravita.ALTO}
    kb.carica_scenario([_incidente(i) for i in ordine], {},
                       {i: (Vulnerabilita.BASSA, 0) for i in ordine}, [("p1", "p2")],
                       gravita_stimata={i: stime[i] for i in ordine})
    assert kb.esposto("p2") is True
    assert kb.gravita_operativa("p2") == "critico"
    assert ("rinforzo", "esposto_a(p1)") in kb.motivi("p2")


def test_campi_ignoti_trattati_in_modo_prudente(kb):
    """Persone ignote valgono come confermate, sostanze ignote come presenti; None non diventa una variabile."""
    inc = [dict(id="q1", tipo=TipoEdificio.RESIDENZIALE, piano=5, sostanze=None, persone=None, meteo=None)]
    kb.carica_scenario(inc, {"cnd": TipoMezzo.CANADAIR, "nbcr": TipoMezzo.NUCLEO_NBCR}, {"q1": (Vulnerabilita.BASSA, 0)},
                       [], {"q1": VelocitaFuoco.BASSA})
    assert kb.gravita("q1") == "alto"  # 2 (persone ignote = confermate) + 0 + 1 (sostanze ignote) = 3
    assert kb.squadra("q1") == ["estinzione", "soccorso_quota", "decontaminazione"]
    assert kb.mezzi_adatti("q1", "decontaminazione") == ["nbcr"]


def test_valore_non_atomico_rifiutato(kb):
    with pytest.raises(ValueError):
        kb.carica_scenario([_incidente("z1", persone="Confermate")], {}, {"z1": (Vulnerabilita.BASSA, 0)}, [])


def test_abduzione_assume_i_campi_ignoti(kb):
    """Con persone ignote la spiegazione assume anche il valore delle persone; ogni spiegazione è minima."""
    inc = [dict(id="a1", tipo=TipoEdificio.INDUSTRIALE, piano=1, sostanze=SostanzePericolose.NO, persone=None,
                meteo=Meteo.CALMO)]
    kb.carica_scenario(inc, {}, {"a1": (Vulnerabilita.ALTA, 2)}, [], {"a1": VelocitaFuoco.BASSA})
    spiegazioni = kb.spiegazioni([("a1", Gravita.CRITICO)])
    assert spiegazioni and all(len(h) == 2 for h in spiegazioni)
    assert ["pers(a1,confermate)", "vel(a1,media)"] in spiegazioni
    assert ["pers(a1,assenti)", "vel(a1,alta)"] not in spiegazioni   # punteggio 4: alto, non critico


def test_osservazioni_incoerenti_non_hanno_spiegazioni(kb):
    """Due fonti che danno gravità diverse allo stesso incidente richiederebbero due velocità: il vincolo
    di integrità lo esclude. Con osservazioni concordi le spiegazioni ci sono."""
    kb.carica_scenario([_incidente("o1", persone=Persone.CONFERMATE)], {}, {"o1": (Vulnerabilita.BASSA, 0)}, [],
                       {"o1": VelocitaFuoco.BASSA})
    assert kb.spiegazioni([("o1", Gravita.ALTO), ("o1", Gravita.CRITICO)]) == []
    assert kb.spiegazioni([("o1", Gravita.ALTO), ("o1", Gravita.ALTO)]) == [["vel(o1,alta)"], ["vel(o1,media)"]]


def test_conflitti_minimi_della_proposta(kb):
    inc = [_incidente("v1", TipoEdificio.RESIDENZIALE, 4, persone=Persone.CONFERMATE),
           _incidente("v2", TipoEdificio.BOSCO, 0)]
    mezzi = {"m_scala": TipoMezzo.AUTOSCALA, "m_pompa": TipoMezzo.AUTOPOMPA, "m_cnd": TipoMezzo.CANADAIR}
    kb.carica_scenario(inc, mezzi, {i["id"]: (Vulnerabilita.BASSA, 0) for i in inc}, [],
                       {i["id"]: VelocitaFuoco.BASSA for i in inc})
    proposta = [("m_scala", "v1", "soccorso_quota"), ("m_pompa", "v1", "estinzione"),
                ("m_pompa", "v2", "estinzione"), ("m_cnd", "v1", "estinzione")]
    conflitti = kb.conflitti(proposta)
    # Il canadair non adatto all'edificio è da solo un conflitto: il conflitto "ruolo coperto due volte"
    # (canadair e autopompa su v1) lo contiene, non è minimo e viene scartato.
    assert conflitti == [["ass(m_cnd,v1,estinzione)"],
                         ["ass(m_pompa,v1,estinzione)", "ass(m_pompa,v2,estinzione)"]]
    assert kb.conflitti([("m_scala", "v1", "soccorso_quota"), ("m_pompa", "v1", "estinzione")]) == []
