# Bioactivity API

A trained classifier served behind a FastAPI endpoint and packaged in a Docker container.
Send it a molecule as a SMILES string, and it returns whether the compound is likely to be
active against the glucocorticoid receptor, with a probability and the descriptors the model saw.

## The model

Trained on a ChEMBL bioactivity export for the glucocorticoid receptor (CHEMBL2034): 3,130
compounds with a pChEMBL value, of which 2,660 are active (pChEMBL 6 or above, meaning an IC50
of 1 micromolar or better) and 470 inactive.

Ten molecular descriptors are computed from each SMILES with RDKit: the four Lipinski
descriptors plus polar surface area, rotatable bonds, aromatic rings, fraction of sp3 carbons,
heavy atom count and ring count. An RBF support vector machine with class weighting sits behind
a standard scaler in a scikit-learn pipeline.

Held-out AUC: **0.824**. The four Lipinski descriptors alone give 0.748, so shape and polarity
are doing real work.

## Running it

Locally:

```bash
pip install -r requirements.txt
python train_model.py          # writes model.joblib
uvicorn main:app --reload
```

In a container:

```bash
docker build -t bioactivity-api .
docker run -p 8000:8000 bioactivity-api
```

Interactive documentation is at http://localhost:8000/docs.

## Endpoints

`GET /health` returns the service status and whether the model loaded. The container health
check calls it.

`POST /predict` takes `{"smiles": "..."}`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO"}'
```

```json
{
  "smiles": "C[C@@H]1C[C@H]2...",
  "active": true,
  "probability": 0.86,
  "descriptors": {"molecular_weight": 392.467, "logp": 1.896, "tpsa": 94.83, "...": "..."}
}
```

That molecule is dexamethasone, a glucocorticoid drug, and the model calls it active at 0.86.
Aspirin comes back inactive at 0.418. A string RDKit cannot parse is rejected with a 422 rather
than a crash.

## Notes

The model is loaded once at start-up rather than per request. Dependencies are pinned and
installed before the code is copied, so editing the code does not trigger a reinstall on rebuild.
