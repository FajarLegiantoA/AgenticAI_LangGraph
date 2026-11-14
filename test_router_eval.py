"""Test akurasi router semantik.

Router adalah satu-satunya bagian proyek yang kualitasnya bisa diukur, jadi
dataset evaluasinya ikut dijaga oleh test suite. Angka ambang batasnya
dikalibrasi pada dataset DEV; dataset HELDOUT tidak pernah dipakai saat
kalibrasi, sehingga test ini benar-benar mengukur generalisasi.

Model embedding dimuat sekali per sesi test (fixture ``scope="session"``)
karena memuatnya butuh belasan detik.
"""

from __future__ import annotations

import pytest

import router
import router_eval


@pytest.fixture(scope="session")
def dev_report() -> router_eval.EvalReport:
    return router_eval.run_evaluation(router_eval.DEV_CASES)


@pytest.fixture(scope="session")
def heldout_report() -> router_eval.EvalReport:
    return router_eval.run_evaluation(router_eval.HELDOUT_CASES)


def test_routing_samples_cover_every_route():
    routes = {route for _, route in router.ROUTING_SAMPLES}

    assert routes == set(router.ROUTES)

    for route in router.ROUTES:
        count = sum(1 for _, name in router.ROUTING_SAMPLES if name == route)

        # Satu sampel saja membuat rute sangat rapuh terhadap jarak.
        assert count >= 5, f"route {route!r} has only {count} samples"


@pytest.mark.slow
def test_dev_dataset_meets_accuracy_target(dev_report):
    assert dev_report.passed, (
        f"DEV {dev_report.summary()}\n" + "\n".join(dev_report.failures)
    )


@pytest.mark.slow
def test_heldout_dataset_meets_accuracy_target(heldout_report):
    assert heldout_report.passed, (
        f"HELDOUT {heldout_report.summary()}\n"
        + "\n".join(heldout_report.failures)
    )


@pytest.mark.slow
def test_out_of_scope_questions_fall_back(dev_report, heldout_report):
    for report in (dev_report, heldout_report):
        assert report.fallback_correct == report.fallback_total


@pytest.mark.slow
def test_app_router_nodes_run_against_real_decision():
    """Guard terhadap salah nama atribut RouteDecision di aplikasi.

    Node router di aplikasi hanya dieksekusi saat graph dijalankan, jadi
    tanpa test ini kesalahan seperti ``decision.matched`` baru ketahuan saat
    user mengetik pertanyaan.
    """
    from langchain_core.messages import HumanMessage

    import app_conditional
    import app_hybrid

    state = {"messages": [HumanMessage(content="How do I fine-tune a model?")]}

    hybrid_update = app_hybrid.router_node(state)
    conditional_update = app_conditional.route_node(state)

    assert hybrid_update["route"] in router.ROUTES
    assert isinstance(hybrid_update["route_distance"], float)
    assert conditional_update["route"] in router.ROUTES

    # System prompt harus tersedia untuk rute mana pun yang terpilih.
    for route in router.ROUTES:
        assert app_hybrid.SYSTEM_PROMPTS[route]


def test_heldout_questions_are_not_duplicates_of_routing_samples():
    """Guard terhadap overfitting: eval tidak boleh menyalin sampel routing."""
    sample_text = {
        text.strip().lower() for _, text in router.ROUTING_SAMPLES
    }

    for question, _ in router_eval.DEV_CASES + router_eval.HELDOUT_CASES:
        assert question.strip().lower() not in sample_text, question
