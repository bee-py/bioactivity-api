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

`POST /predict` takes `{"smiles": "..."}`.

### How to send a molecule

The body must be a JSON object with one field called `smiles`. The braces and quotes matter:

```json
{"smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"}
```

Sending the SMILES on its own, like `"Cn1cnc2c1c(=O)n(C)c(=O)n2C"`, returns **422** before the
molecule is ever read, because there is no `smiles` field to find. This catches people out in
the `/docs` page: replace only the text between the quotation marks and leave the rest of the
line alone.

A SMILES string RDKit cannot parse also returns 422, with a message naming the string, so the
two cases are easy to tell apart from the response.

Three molecules to try, with what the model says about them:

| Molecule | SMILES | Result |
|---|---|---|
| Dexamethasone, a glucocorticoid drug | `C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO` | active, 0.86 |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | inactive, 0.418 |
| Caffeine | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` | inactive, 0.328 |

A full request:

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

## Credits

The task framing comes from my MSc coursework on the glucocorticoid receptor, which took its
lead from two published models for the same target:

- N. Schaduangrat, H. Chuntakaruk, T. Rungrotmongkol, P. Mookdarsanit and W. Shoombuatong,
  *M3S-GRPred*, BMC Bioinformatics, 2025. https://doi.org/10.1186/s12859-025-06132-1
- W. Shoombuatong, P. Mookdarsanit, N. Schaduangrat and L. Mookdarsanit, *BGATT-GR*,
  Scientific Reports, 2025. https://doi.org/10.1038/s41598-025-05839-8

Both use several fingerprint types and report stronger performance than the model here, which
uses ten RDKit descriptors and exists to demonstrate serving and deployment rather than to
compete with them. Neither paper's code is reused.

Method and data sources: C. Cortes and V. Vapnik, *Support-vector networks*, Machine Learning,
1995, for the classifier; J. Platt, *Probabilistic outputs for support vector machines*, 1999,
for turning SVM scores into probabilities, which is what scikit-learn's `probability=True` does;
and D. Mendez et al., *ChEMBL: towards direct deposition of bioassay data*, Nucleic Acids
Research, 2018, for the bioactivity data. Descriptors are computed with RDKit.
