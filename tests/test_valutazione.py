"""Test del giudizio di un piano con la gravità vera (src/valutazione_dispatch.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import valutazione_dispatch as vd  # noqa: E402

COSTO = {(m, i): 2 for m in ("m0", "m1", "m2") for i in ("i0", "i1")}


def test_abbinamento_massimo_non_greedy():
    # m0 adatto a entrambi i ruoli, m1 solo al primo: l'abbinamento massimo copre tutti e due.
    adatti = {("i0", "estinzione"): ["m0", "m1"], ("i0", "rinforzo"): ["m0"]}
    assert vd._abbinamento(["m1", "m0"], list(adatti), adatti) == 2


def test_piano_corretto_non_ha_scoperti_ne_superflui():
    veri = {("i0", "estinzione"): ["m0", "m1"], ("i1", "estinzione"): ["m2"]}
    piano = [("m0", "i0", "estinzione"), ("m2", "i1", "estinzione")]
    esito = vd.giudica(piano, veri, COSTO, penalita=10)
    assert esito == dict(ruoli_veri_scoperti=0, mezzi_superflui=0, costo_valutato=4)


def test_gravita_sottostimata_lascia_ruolo_scoperto():
    # Il rinforzo vero non è stato chiesto: un ruolo scoperto, pagato con la penalità.
    veri = {("i0", "estinzione"): ["m0", "m1"], ("i0", "rinforzo"): ["m0", "m1"]}
    esito = vd.giudica([("m0", "i0", "estinzione")], veri, COSTO, penalita=10)
    assert esito["ruoli_veri_scoperti"] == 1 and esito["costo_valutato"] == 12


def test_gravita_sovrastimata_manda_un_mezzo_superfluo():
    veri = {("i0", "estinzione"): ["m0", "m1"]}
    piano = [("m0", "i0", "estinzione"), ("m1", "i0", "rinforzo")]
    esito = vd.giudica(piano, veri, COSTO, penalita=10)
    assert esito["mezzi_superflui"] == 1 and esito["ruoli_veri_scoperti"] == 0


def test_completamento_conta_solo_per_gli_incidenti_veramente_critici():
    veri = {("i0", "estinzione"): ["m0", "m1"], ("i0", "rinforzo"): ["m0", "m1"]}
    costo = {("m0", "i0"): 2, ("m1", "i0"): 7}
    piano = [("m0", "i0", "estinzione"), ("m1", "i0", "rinforzo")]
    assert vd.giudica(piano, veri, costo, penalita=10)["costo_valutato"] == 9
    assert vd.giudica(piano, veri, costo, penalita=10, critici_veri=frozenset({"i0"}), peso=1)["costo_valutato"] == 16
