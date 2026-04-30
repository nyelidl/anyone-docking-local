# ── 1. Install dependencies ────────────────────────────────
python ACD_standalone.py setup

# ── 2. Check environment ─────────────────────────────────
python ACD_standalone.py info

# ── 3. Dock ligand [single mode] ──────────────────────────────────
python ACD_standalone.py dock \
    -r 1M17.pdb \
    --smiles "O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12" \
    --name Baicalein --ph 7.4 --mode pkanet

# Auto-download PDB from RCSB (PDB ID)
python ACD_standalone.py dock -r 1M17 --smiles "CCO" --name Ethanol

# ── 4. Batch docking from .smi file ────────────────────────
python ACD_standalone.py batch \
    -r 1M17.pdb --smi ligands.smi \
    --exhaustiveness 16 -o ./results
