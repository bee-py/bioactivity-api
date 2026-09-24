"""Train the glucocorticoid receptor activity classifier and save it for the API.

Data: ChEMBL bioactivity export for the glucocorticoid receptor (CHEMBL2034).
Label: a compound is active when pChEMBL >= 6 (IC50 of 1 micromolar or better).
Features: the four Lipinski descriptors, computed from the SMILES with RDKit, so the
API can take a SMILES string and work out the numbers itself.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RDLogger.DisableLog("rdApp.*")

DATA = Path(r"C:\Users\beula\OneDrive\Documents\MSc Digi Chem\AI Drug Discovery\Assessment\bioactivitydata.csv")
OUT = Path(__file__).parent / "model.joblib"
FEATURES = ["molecular_weight", "logp", "h_donors", "h_acceptors", "tpsa", "rotatable_bonds",
            "aromatic_rings", "fraction_csp3", "heavy_atoms", "rings"]


def descriptors(smiles: str):
    """Molecular descriptors, or None if RDKit cannot read the molecule.

    The first four are the Lipinski set; the rest add shape and polarity, which separate
    the actives better than size and greasiness alone.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "h_donors": Lipinski.NumHDonors(mol),
        "h_acceptors": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "rings": Lipinski.RingCount(mol),
    }


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep=";", low_memory=False)
    df = df[df["pChEMBL Value"].notna() & df["Smiles"].notna()]
    df = df.drop_duplicates(subset="Smiles")

    rows = []
    for smiles, pchembl in zip(df["Smiles"], df["pChEMBL Value"]):
        d = descriptors(smiles)
        if d is None:
            continue
        d["active"] = int(pchembl >= 6)
        rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    frame = load_training_frame()
    print(f"{len(frame)} compounds, {frame['active'].sum()} active, "
          f"{len(frame) - frame['active'].sum()} inactive")

    X = frame[FEATURES].to_numpy(dtype=float)
    y = frame["active"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True,
                    random_state=42)),
    ])
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"test AUC {roc_auc_score(y_test, probabilities):.3f}")

    joblib.dump({"model": model, "features": FEATURES}, OUT)
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
