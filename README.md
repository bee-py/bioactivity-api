# Bioactivity API

A model that predicts whether a compound is active against the glucocorticoid receptor, running
as a web service. You send a molecule as a SMILES string and get back a probability, a yes or
no, and the descriptors the model used.

## The model

The training data is a ChEMBL export for the glucocorticoid receptor (CHEMBL2034). I kept the
3,130 compounds that have a pChEMBL value, and labelled a compound active if that value is 6 or
above, which is an IC50 of 1 micromolar or better. That gives 2,660 active and 470 inactive.

RDKit computes ten descriptors for each compound from its SMILES: molecular weight, logP,
hydrogen bond donors and acceptors, polar surface area, rotatable bonds, aromatic rings,
fraction of sp3 carbons, heavy atoms and rings. The descriptors are scaled, then a support
vector machine with an RBF kernel is trained on them. Classes are weighted, because there are
about five times more actives than inactives.

The AUC on the held-out 20% is **0.824**. Using only the first four descriptors gives 0.748.

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
molecule is ever read, because there is no `smiles` field to find. In the `/docs` page, replace
only the text between the quotation marks and leave the rest of the line alone.

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

Both use several fingerprint types and report better results than this model, which uses ten
RDKit descriptors. I built it to practise serving and deployment, not to match their numbers.
No code from either paper is used here.

Method and data sources: C. Cortes and V. Vapnik, *Support-vector networks*, Machine Learning,
1995, for the classifier; J. Platt, *Probabilistic outputs for support vector machines*, 1999,
for turning SVM scores into probabilities; and D. Mendez et al., *ChEMBL: towards direct deposition of bioassay data*, Nucleic Acids
Research, 2018, for the bioactivity data. Descriptors are computed with RDKit.
