"""Tests for the bioactivity service.

Run with pytest from this folder. The tests call the app in memory, so the service does not
need to be running first.
"""
import pytest
from fastapi.testclient import TestClient

from main import FEATURES, THRESHOLD, app

DEXAMETHASONE = "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

client = TestClient(app)


def post(smiles: str):
    return client.post("/predict", json={"smiles": smiles})


def test_health_says_the_model_is_loaded():
    """Checks the health endpoint reports a loaded model and all ten measurements. A service
    can be running and still be useless if the model file did not load."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "features": len(FEATURES)}


@pytest.mark.parametrize("smiles, expected_active", [(DEXAMETHASONE, True), (CAFFEINE, False)])
def test_the_two_drugs_come_out_the_right_way_round(smiles, expected_active):
    """Checks dexamethasone comes back active and caffeine comes back inactive. Both are
    needed, because a model that called everything active would pass on dexamethasone alone."""
    assert post(smiles).json()["active"] is expected_active


def test_the_answer_contains_everything_it_should():
    """Checks the answer has the ten measurements, a probability between 0 and 1, and a yes or
    no that matches that probability. The last part means the yes or no can never disagree with
    the probability it came from."""
    body = post(DEXAMETHASONE).json()
    assert set(body["descriptors"]) == set(FEATURES)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["active"] == (body["probability"] >= THRESHOLD)


@pytest.mark.parametrize("body, named_in_message", [
    ({"smiles": "not-a-molecule"}, "not-a-molecule"),
    (CAFFEINE, "valid dictionary"),
])
def test_bad_input_is_refused_with_a_reason(body, named_in_message):
    """Checks an unreadable molecule, and a request sent without the smiles field, both return
    422 with a message saying what was wrong. The two are refused in different places: RDKit
    cannot read the first, and FastAPI rejects the second before this code runs."""
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert named_in_message in str(r.json()["detail"])
