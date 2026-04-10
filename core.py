#!/usr/bin/env python3
"""
core.py — Pure computation layer for Anyone Can Dock.
No Streamlit imports. All functions return plain dicts / tuples.
Safe to import in Colab notebooks, pytest, or any UI framework.

Additions / fixes vs previous version
──────────────────────────────────────
• prepare_ligand_from_file  — was missing; needed for uploaded-structure mode
• _parse_smiles_robust      — extracted to module level; shared by all diagram fns
• _classify_interaction_full — richer 8-type classifier replacing simple 3-type one
• _min_dist_residue         — helper used throughout diagram code
• draw_interactions_rdkit_classic — classic RDKit highlight-circle diagram (Tab 2)
• draw_interaction_diagram_data   — data dict for the interactive JS widget (Tab 1)
• draw_interaction_diagram        — static SVG version of the custom diagram (Tab 1)
• _get_lig_svg_and_atom_coords    — helper: render ligand + get pixel coords
• _place_residue_bubbles          — helper: radial bubble placement algorithm
• draw_interactions_rdkit now uses module-level _parse_smiles_robust
• fix_sdf_bond_orders: never adds H atoms without 3-D coords (PoseView rejection fix)
• call_poseview_v1 / call_poseview2_ref: 3 retries, full response logging
"""

import os
import re as _re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

METAL_RESNAMES = {
    "MG", "ZN", "CA", "MN", "FE", "CU", "CO", "NI", "CD", "HG", "NA", "K",
}
METAL_CHARGES = {
    "MG": 2.0, "ZN": 2.0, "CA": 2.0, "MN": 2.0, "FE": 3.0,
    "CU": 2.0, "CO": 2.0, "NI": 2.0, "CD": 2.0, "HG": 2.0,
    "NA": 1.0, "K":  1.0,
}

EXCLUDE_IONS = set(
    "HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG".split(",")
)
GLYCAN_NAMES = {
    "NAG", "BMA", "MAN", "FUC", "GAL", "GLC", "SIA", "NGA",
    "FUL", "GLA", "BGC", "A2G", "LAT", "MAL", "CEL", "SUC",
    "TRE", "GCS", "NDG", "NGC",
}
COFACTOR_NAMES = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP",
    "NAD", "NAP", "NDP", "FAD", "FMN",
    "HEM", "HEC", "HEA",
    "GOL", "PEG", "EDO", "MPD", "PGE", "PG4",
    "SO4", "PO4", "SUL", "PHO",
    "IHP", "TTP", "CTP", "UTP",
    "COA", "SAM", "SAH",
    "EPE", "MES", "TRS", "ACT", "ACY",
}

_PV_MAX_RETRIES  = 3
_PV_RETRY_DELAY  = 10
_PV_POLL_ATTEMPTS = 60

# ── Interaction diagram rendering constants ───────────────────────────────────
_DIAG_W = 900
_DIAG_H = 760

_DIAG_TYPE_CFG: dict = {
    "hbond":            {"fill": "#80dd80", "stroke": "#1a7a1a",
                         "lineclr": "#1a7a1a",  "dash": "5 3",     "lw": "1.6"},
    "hbond_to_halogen": {"fill": "#c4a0ff", "stroke": "#6633aa",
                         "lineclr": "#6633aa",  "dash": "4 2 1 2", "lw": "1.6"},
    "pi_pi":            {"fill": "#f0a0ff", "stroke": "#e200e8",
                         "lineclr": "#e200e8",  "dash": "5 3",     "lw": "1.6"},
    "cation_pi":        {"fill": "#f0a0ff", "stroke": "#e200e8",
                         "lineclr": "#e200e8",  "dash": "5 3",     "lw": "1.6"},
    "hydrophobic":      {"fill": "#a0c8ff", "stroke": "#2287ff",
                         "lineclr": None,        "dash": "",        "lw": "0"},
    "ionic":            {"fill": "#ffb0d0", "stroke": "#cc2277",
                         "lineclr": "#cc2277",  "dash": "6 2 2 2", "lw": "1.8"},
    "metal":            {"fill": "#ffe090", "stroke": "#cc8800",
                         "lineclr": "#cc8800",  "dash": "3 2",     "lw": "1.8"},
    "halogen":          {"fill": "#ffb0d0", "stroke": "#cc2277",
                         "lineclr": "#cc2277",  "dash": "5 2",     "lw": "1.6"},
}

# Classic-tab 3-colour palette
_CLASSIC_TYPE_COLOR: dict = {
    "hbond":            (0.357, 0.608, 0.835),   # blue
    "hbond_to_halogen": (0.545, 0.306, 0.835),   # purple
    "ionic":            (0.851, 0.263, 0.533),   # pink-red
    "pi_pi":            (0.851, 0.263, 0.851),   # magenta
    "cation_pi":        (0.851, 0.263, 0.851),   # magenta
    "metal":            (0.851, 0.620, 0.110),   # amber
    "hydrophobic":      (0.173, 0.545, 0.341),   # green
    "halogen":          (0.851, 0.263, 0.533),   # pink-red
}
_CLASSIC_DEFAULT_CLR = (0.800, 0.373, 0.541)    # pink — other


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd, cwd=None):
    """Run a shell command. Returns (returncode, combined_stdout_stderr)."""
    r = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def _rdkit_six_patch():
    try:
        from rdkit import six  # noqa
    except ImportError:
        from io import StringIO as _SIO
        from types import ModuleType as _MT
        import rdkit as _rdkit
        _m = _MT("six")
        _m.StringIO = _SIO
        _m.PY3 = True
        _rdkit.six = _m
        sys.modules["rdkit.six"] = _m


def _strip_h_from_pdb(pdb_path: str, out_path: str) -> bool:
    import shutil
    try:
        lines = []
        with open(pdb_path) as f:
            for line in f:
                rec = line[:6].strip()
                if rec in ("ATOM", "HETATM"):
                    atom_name = line[12:16].strip()
                    element   = line[76:78].strip() if len(line) > 76 else ""
                    if element.upper() == "H" or atom_name.startswith("H"):
                        continue
                lines.append(line)
        with open(out_path, "w") as f:
            f.writelines(lines)
        return True
    except Exception:
        shutil.copy(pdb_path, out_path)
        return False


def convert_cif_to_pdb(cif_path: str, pdb_out_path: str) -> dict:
    log = []

    try:
        import gemmi
        doc   = gemmi.cif.read(cif_path)
        block = doc.sole_block()
        st    = gemmi.make_structure_from_block(block)
        st.setup_entities()
        st.assign_label_seq_id()
        pdb_str = st.make_pdb_headers() + st.make_pdb_string()
        with open(pdb_out_path, "w") as f:
            f.write(pdb_str)
        if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
            log.append("✓ CIF → PDB via gemmi")
            return {"success": True, "pdb_path": pdb_out_path, "log": log}
        log.append("⚠ gemmi produced empty PDB — trying OpenBabel")
    except ImportError:
        log.append("⚠ gemmi not installed — trying OpenBabel")
    except Exception as e:
        log.append(f"⚠ gemmi failed ({e}) — trying OpenBabel")

    try:
        rc, out = run_cmd(f'obabel "{cif_path}" -O "{pdb_out_path}"')
        if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
            log.append("✓ CIF → PDB via OpenBabel")
            return {"success": True, "pdb_path": pdb_out_path, "log": log}
        log.append(f"⚠ OpenBabel produced empty file (exit {rc}): {out[:200]}")
    except Exception as e:
        log.append(f"⚠ OpenBabel failed: {e}")

    try:
        from prody import parseMMCIF, writePDB as _writePDB
        atoms = parseMMCIF(cif_path)
        if atoms is not None and atoms.numAtoms() > 0:
            _writePDB(pdb_out_path, atoms)
            if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
                log.append("✓ CIF → PDB via ProDy")
                return {"success": True, "pdb_path": pdb_out_path, "log": log}
        log.append("⚠ ProDy produced empty PDB")
    except ImportError:
        log.append("⚠ ProDy parseMMCIF not available")
    except Exception as e:
        log.append(f"⚠ ProDy failed: {e}")

    return {
        "success": False, "pdb_path": pdb_out_path, "log": log,
        "error": "All CIF→PDB methods failed. Install gemmi: pip install gemmi",
    }


def is_cif_file(filepath: str) -> bool:
    ext = Path(filepath).suffix.lower()
    if ext in (".cif", ".mmcif"):
        return True
    try:
        with open(filepath) as f:
            first = f.read(512)
        if first.strip().startswith("data_"):
            return True
    except Exception:
        pass
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SMILES PARSING (module level — shared by all diagram functions)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_smiles_robust(smi: str):
    """
    Parse SMILES preserving formal charges ([O-], [NH3+], etc.).
    Handles aromatic-ketone notation that RDKit's direct parser rejects.
    Returns RDKit mol or None.
    """
    if not smi:
        return None
    from rdkit import Chem
    # A: direct
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return m
    # B: fix aromatic-ketone / non-Kekulé aromatic
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            return None
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
        Chem.SetAromaticity(m)
        canon = Chem.MolToSmiles(m)
        m2 = Chem.MolFromSmiles(canon)
        if m2 is not None:
            return m2
        try:
            Chem.SanitizeMol(m, catchErrors=True)
        except Exception:
            pass
        return m
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL AVAILABILITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_obabel():
    import shutil
    if shutil.which("obabel") is None:
        return False, "obabel not found — add 'openbabel' to packages.txt"
    _, out = run_cmd("obabel --version")
    return True, (out.splitlines()[0] if out else "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  VINA BINARY
# ══════════════════════════════════════════════════════════════════════════════

def get_vina_binary(path: str = ""):
    import platform
    system  = platform.system().lower()
    machine = platform.machine().lower()
    _BASE   = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/"

    if system == "linux":
        _FNAME = "vina_1.2.7_linux_x86_64"
    elif system == "darwin":
        _FNAME = ("vina_1.2.7_mac_aarch64"
                  if machine in ("arm64", "aarch64")
                  else "vina_1.2.7_mac_x86_64")
    elif system == "windows":
        _FNAME = "vina_1.2.7_windows_x86_64.exe"
    else:
        return None, f"Unsupported platform: {system}/{machine}"

    _URL = _BASE + _FNAME
    if not path:
        path = os.path.join(tempfile.gettempdir(), _FNAME)

    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        try:
            import urllib.request
            urllib.request.urlretrieve(_URL, path)
        except Exception as e1:
            try:
                import requests
                r = requests.get(_URL, stream=True, timeout=120)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            except Exception as e2:
                return None, f"Download failed: {e1} / {e2}"
    if system != "windows":
        os.chmod(path, 0o755)
    return path, f"ok ({system}/{machine})"


# ══════════════════════════════════════════════════════════════════════════════
#  RECEPTOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def detect_cocrystal_ligand(raw_pdb: str) -> dict:
    from prody import parsePDB, calcCenter
    atoms = parsePDB(raw_pdb)
    if atoms is None:
        return {"found": False}
    excl = EXCLUDE_IONS | GLYCAN_NAMES | COFACTOR_NAMES
    het  = atoms.select("hetero and not water")
    if het is None:
        return {"found": False}
    cands = [
        r for r in het.getHierView().iterResidues()
        if (r.getResname() or "").strip() not in excl
    ]
    if not cands:
        return {"found": False}
    cands.sort(key=lambda r: (-r.numAtoms(), r.getChid() != "A"))
    chosen = cands[0]
    rn, ch, ri = chosen.getResname(), chosen.getChid(), chosen.getResnum()
    sel_str    = f"resname {rn} and resid {ri} and chain {ch}"
    lig_atoms  = atoms.select(sel_str)
    cx, cy, cz = (float(v) for v in calcCenter(lig_atoms))
    return {
        "found": True, "resname": rn, "chain": ch, "resid": ri,
        "sel_str": sel_str, "ligand_id": f"{rn}_{ch}_{ri}",
        "cx": cx, "cy": cy, "cz": cz,
        "n_atoms": lig_atoms.numAtoms(), "atoms": lig_atoms,
    }


def strip_and_convert_receptor(rec_raw: str, wdir) -> dict:
    wdir   = Path(wdir)
    log    = []
    rec_fh    = str(wdir / "rec.pdb")
    rec_pdbqt = str(wdir / "rec.pdbqt")
    try:
        metal_lines, clean_lines = [], []
        with open(rec_raw) as f:
            for line in f:
                field = line[:6].strip()
                if (field in ("ATOM", "HETATM")
                        and line[17:20].strip().upper() in METAL_RESNAMES):
                    metal_lines.append(line)
                else:
                    clean_lines.append(line)
        rec_nometal = str(wdir / "receptor_atoms_nometal.pdb")
        with open(rec_nometal, "w") as f:
            f.writelines(clean_lines)
        if metal_lines:
            names = ", ".join(sorted({l[17:20].strip() for l in metal_lines}))
            log.append(f"⚠ Stripped {len(metal_lines)} metal atoms before OpenBabel: {names}")

        rc1, out1 = run_cmd(f'obabel "{rec_nometal}" -O "{rec_fh}" -h')
        if not os.path.exists(rec_fh) or os.path.getsize(rec_fh) < 100:
            raise ValueError(f"OpenBabel H-addition empty (exit {rc1}): {out1[:400]}")
        log.append("✓ Hydrogens added")

        rc2, out2 = run_cmd(
            f'obabel "{rec_fh}" -O "{rec_pdbqt}" -xr --partialcharge gasteiger'
        )
        if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
            raise ValueError(f"PDBQT conversion empty (exit {rc2}): {out2[:400]}")
        log.append("✓ PDBQT conversion complete")

        if metal_lines:
            pdbqt_lines = [l for l in open(rec_pdbqt).readlines()
                           if l.strip() != "END"]
            injected = 0
            for ml in metal_lines:
                try:
                    resname = ml[17:20].strip().upper()
                    serial  = int(ml[6:11])
                    name    = ml[12:16].strip()
                    chain   = ml[21] if len(ml) > 21 else "A"
                    resid   = int(ml[22:26])
                    x, y, z = float(ml[30:38]), float(ml[38:46]), float(ml[46:54])
                    charge  = METAL_CHARGES.get(resname, 0.0)
                    atype   = resname.capitalize()
                    pdbqt_lines.append(
                        f"HETATM{serial:5d} {name:<4s} {resname:<3s} "
                        f"{chain}{resid:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                        f"    {charge:+.3f} {atype}\n"
                    )
                    injected += 1
                except Exception as e:
                    log.append(f"⚠ Could not re-inject metal: {e}")
            pdbqt_lines.append("END\n")
            with open(rec_pdbqt, "w") as f:
                f.writelines(pdbqt_lines)
            log.append(f"✅ Re-injected {injected} metal atom(s) into PDBQT")

        log.append("✓ Receptor PDBQT ready")
        return {"success": True, "rec_fh": rec_fh, "rec_pdbqt": rec_pdbqt, "log": log}
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


def write_box_pdb(filename, cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners = [
        (cx + dx, cy + dy, cz + dz)
        for dx in (-hx, hx) for dy in (-hy, hy) for dz in (-hz, hz)
    ]
    with open(filename, "w") as f:
        for i, (x, y, z) in enumerate(corners, 1):
            f.write(f"HETATM{i:5d}  C   BOX A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("CONECT    1    2    3    5\nCONECT    2    1    4    6\n"
                "CONECT    3    1    4    7\nCONECT    4    2    3    8\n"
                "CONECT    5    1    6    7\nCONECT    6    2    5    8\n"
                "CONECT    7    3    5    8\nCONECT    8    4    6    7\n")


def write_vina_config(filename, cx, cy, cz, sx, sy, sz):
    with open(filename, "w") as f:
        f.write(f"center_x = {cx:.4f}\ncenter_y = {cy:.4f}\ncenter_z = {cz:.4f}\n"
                f"size_x = {sx}\nsize_y = {sy}\nsize_z = {sz}\n")


def prepare_receptor(
    raw_pdb: str, wdir,
    center_mode: str = "auto",
    manual_xyz: tuple = (0.0, 0.0, 0.0),
    prody_sel: str = "",
    box_size: tuple = (16, 16, 16),
) -> dict:
    from prody import parsePDB, calcCenter, writePDB
    wdir = Path(wdir)
    log  = []
    sx, sy, sz = box_size
    try:
        if is_cif_file(raw_pdb):
            log.append("📄 Detected mmCIF — converting to PDB…")
            converted = str(wdir / "converted_from_cif.pdb")
            cif_res   = convert_cif_to_pdb(raw_pdb, converted)
            log.extend(cif_res["log"])
            if not cif_res["success"]:
                raise ValueError(f"CIF→PDB failed: {cif_res.get('error')}")
            raw_pdb = converted

        atoms = parsePDB(raw_pdb)
        if atoms is None:
            raise ValueError("ProDy parsePDB returned None")
        log.append(f"✓ Parsed {atoms.numAtoms()} atoms")

        ligand_pdb_path = None
        ligand_sel_str  = None
        cocrystal_ligand_id = ""
        rn = ch = ""
        ri = 0
        cx = cy = cz = 0.0

        if center_mode == "auto":
            info = detect_cocrystal_ligand(raw_pdb)
            if info["found"]:
                rn, ch, ri = info["resname"], info["chain"], info["resid"]
                ligand_sel_str      = info["sel_str"]
                cocrystal_ligand_id = info["ligand_id"]
                cx, cy, cz          = info["cx"], info["cy"], info["cz"]
                ligand_pdb_path     = str(wdir / "LIG.pdb")
                writePDB(ligand_pdb_path, info["atoms"])
                log.append(f"✓ Co-crystal: {rn} chain {ch} resnum {ri} ({info['n_atoms']} atoms)")
                log.append(f"📍 Center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")
            else:
                log.append("⚠ No co-crystal ligand found after filtering")

        elif center_mode == "manual":
            cx, cy, cz = (float(v) for v in manual_xyz)
            log.append(f"🛠 Manual center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")

        elif center_mode == "selection":
            if not prody_sel.strip():
                raise ValueError("ProDy selection string is empty.")
            ref_atoms = atoms.select(prody_sel.strip())
            if ref_atoms is None or ref_atoms.numAtoms() == 0:
                raise ValueError(f"ProDy selection '{prody_sel}' matched 0 atoms.")
            cx, cy, cz = (float(v) for v in calcCenter(ref_atoms))
            log.append(f"🔬 ProDy '{prody_sel}' → {ref_atoms.numAtoms()} atoms")
            log.append(f"📍 Center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            _resnames = list(dict.fromkeys(ref_atoms.getResnames()))
            _resids   = list(dict.fromkeys(ref_atoms.getResnums()))
            _chains   = list(dict.fromkeys(ref_atoms.getChids()))
            if len(_resnames) == 1 and len(_resids) == 1:
                rn, ri, ch = _resnames[0], int(_resids[0]), (_chains[0] if _chains else "A")
                ligand_sel_str      = f"resname {rn} and resid {ri} and chain {ch}"
                cocrystal_ligand_id = f"{rn}_{ch}_{ri}"
                ligand_pdb_path     = str(wdir / "LIG.pdb")
                writePDB(ligand_pdb_path, ref_atoms)
                log.append(f"✓ Ligand: {rn} chain {ch} resnum {ri}")
                log.append(f"🔑 PoseView2 ID: {cocrystal_ligand_id}")
            else:
                ligand_pdb_path = str(wdir / "LIG_ref.pdb")
                writePDB(ligand_pdb_path, ref_atoms)
                log.append("⚠ Multi-residue selection — PoseView2 ID not set")

        sel_str = (f"not ({ligand_sel_str}) and not water"
                   if ligand_sel_str else "not water")
        rec_sel = atoms.select(sel_str)
        if rec_sel is None or rec_sel.numAtoms() == 0:
            raise ValueError("Receptor selection returned no atoms")
        rec_raw_path = str(wdir / "receptor_atoms.pdb")
        writePDB(rec_raw_path, rec_sel)
        log.append(f"✓ Receptor: {rec_sel.numAtoms()} atoms")

        conv = strip_and_convert_receptor(rec_raw_path, wdir)
        log.extend(conv["log"])
        if not conv["success"]:
            raise ValueError(conv["error"])

        box_pdb  = str(wdir / "rec.box.pdb")
        cfg_path = str(wdir / "rec.box.txt")
        write_box_pdb(box_pdb, cx, cy, cz, sx, sy, sz)
        write_vina_config(cfg_path, cx, cy, cz, sx, sy, sz)
        log.append("✓ Box + config written")

        return {
            "success": True,
            "rec_fh":  conv["rec_fh"], "rec_pdbqt": conv["rec_pdbqt"],
            "box_pdb": box_pdb, "config_txt": cfg_path,
            "cx": cx, "cy": cy, "cz": cz,
            "sx": sx, "sy": sy, "sz": sz,
            "ligand_pdb_path":     ligand_pdb_path,
            "cocrystal_ligand_id": cocrystal_ligand_id,
            "n_atoms": rec_sel.numAtoms(), "log": log,
        }
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


# ══════════════════════════════════════════════════════════════════════════════
#  LIGAND PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _meeko_to_pdbqt(mol, out_path: str):
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    try:
        from meeko import PDBQTWriterLegacy
        setups    = prep.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(setups[0])
    except (ImportError, AttributeError):
        prep.prepare(mol)
        pdbqt_str = prep.write_pdbqt_string()
    with open(out_path, "w") as f:
        f.write(pdbqt_str)


def prepare_ligand(smiles: str, name: str, ph: float, wdir) -> dict:
    """Protonate at target pH → 3D conformer → Meeko PDBQT + SDF."""
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    wdir      = Path(wdir)
    log       = []
    out_pdbqt = str(wdir / f"{name}.pdbqt")
    out_sdf   = str(wdir / f"{name}_3d.sdf")
    try:
        prot = smiles.strip()
        try:
            from dimorphite_dl import protonate_smiles
            vs = protonate_smiles(prot, ph_min=ph, ph_max=ph, max_variants=1)
            if vs:
                prot = vs[0]
                log.append(f"✓ Dimorphite-DL pH {ph}")
        except Exception as e:
            log.append(f"⚠ Dimorphite-DL skipped: {e}")

        mol = Chem.MolFromSmiles(prot)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {prot[:60]}")

        charge = Chem.GetFormalCharge(mol)
        log.append(f"✓ Formal charge: {charge:+d}")

        mol = Chem.AddHs(mol)
        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        log.append("✓ 3D conformer generated + minimised")

        with Chem.SDWriter(out_sdf) as w:
            w.write(mol)
        _meeko_to_pdbqt(mol, out_pdbqt)
        log.append("✓ PDBQT written (Meeko)")

        return {
            "success": True, "pdbqt": out_pdbqt, "sdf": out_sdf,
            "prot_smiles": prot, "charge": charge, "log": log,
        }
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


def prepare_ligand_from_file(file_path: str, name: str, wdir) -> dict:
    """
    Prepare a ligand from an uploaded SDF / MOL2 / PDB file.
    Keeps molecular form as-is; generates 3D coords only if absent.
    Returns same dict shape as prepare_ligand.
    """
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    wdir      = Path(wdir)
    log       = []
    out_pdbqt = str(wdir / f"{name}.pdbqt")
    out_sdf   = str(wdir / f"{name}_3d.sdf")
    try:
        ext = Path(file_path).suffix.lower()
        mol = None

        if ext == ".sdf":
            supp = Chem.SDMolSupplier(file_path, sanitize=True, removeHs=False)
            mol  = next((m for m in supp if m is not None), None)
        else:
            # Convert via obabel → SDF
            tmp_sdf = str(wdir / f"{name}_upload_conv.sdf")
            run_cmd(f'obabel "{file_path}" -O "{tmp_sdf}" -h 2>/dev/null')
            if os.path.exists(tmp_sdf) and os.path.getsize(tmp_sdf) > 10:
                supp = Chem.SDMolSupplier(tmp_sdf, sanitize=True, removeHs=False)
                mol  = next((m for m in supp if m is not None), None)

        if mol is None:
            raise ValueError(f"Could not parse molecule from {Path(file_path).name}")

        log.append(f"✓ Read molecule: {mol.GetNumAtoms()} atoms")

        # Ensure Hs and 3D conformer
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            try:
                params = AllChem.ETKDGv3()
            except AttributeError:
                params = AllChem.ETKDG()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) == -1:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            log.append("✓ 3D conformer generated + minimised")
        else:
            # Add Hs with coords if missing
            if not any(mol.GetAtomWithIdx(i).GetAtomicNum() == 1
                       for i in range(mol.GetNumAtoms())):
                mol = Chem.AddHs(mol, addCoords=True)
            log.append("✓ Used existing 3D coordinates")

        charge = Chem.GetFormalCharge(mol)
        log.append(f"✓ Formal charge: {charge:+d}")

        mol_noH = Chem.RemoveHs(mol, sanitize=False)
        try:
            Chem.SanitizeMol(mol_noH)
            prot_smiles = Chem.MolToSmiles(mol_noH)
        except Exception:
            prot_smiles = ""

        with Chem.SDWriter(out_sdf) as w:
            w.write(mol)
        _meeko_to_pdbqt(mol, out_pdbqt)
        log.append("✓ PDBQT written (Meeko)")

        return {
            "success": True, "pdbqt": out_pdbqt, "sdf": out_sdf,
            "prot_smiles": prot_smiles, "charge": charge, "log": log,
        }
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


def smiles_from_file(file_path: str, wdir) -> str:
    """Extract SMILES from SDF / MOL2 / PDB. Raises ValueError on failure."""
    wdir = Path(wdir)
    ext  = Path(file_path).suffix.lower()
    if ext == ".sdf":
        from rdkit import Chem
        mols = [m for m in Chem.SDMolSupplier(file_path, sanitize=True) if m]
        if not mols:
            raise ValueError("No valid molecule in SDF")
        return Chem.MolToSmiles(mols[0])
    smi_tmp = str(wdir / "lig_upload.smi")
    run_cmd(f'obabel "{file_path}" -O "{smi_tmp}" --canonical 2>/dev/null')
    for line in open(smi_tmp):
        pts = line.strip().split(None, 1)
        if pts:
            return pts[0]
    raise ValueError("Could not convert structure file to SMILES")


# ══════════════════════════════════════════════════════════════════════════════
#  DOCKING
# ══════════════════════════════════════════════════════════════════════════════

def run_vina(
    receptor_pdbqt, ligand_pdbqt, config_txt, vina_path,
    exhaustiveness=16, n_modes=10, energy_range=3,
    wdir=".", out_name="out",
) -> dict:
    wdir      = Path(wdir)
    out_pdbqt = str(wdir / f"{out_name}_out.pdbqt")
    out_sdf   = str(wdir / f"{out_name}_out.sdf")
    rc, vlog  = run_cmd(
        f'"{vina_path}" '
        f'--receptor "{receptor_pdbqt}" --ligand "{ligand_pdbqt}" '
        f'--config "{config_txt}" '
        f'--exhaustiveness {exhaustiveness} '
        f'--num_modes {n_modes} --energy_range {energy_range} '
        f'--out "{out_pdbqt}"',
        cwd=str(wdir),
    )
    if rc != 0 or not os.path.exists(out_pdbqt):
        return {"success": False, "error": f"Vina exit {rc}", "log": vlog}
    run_cmd(f'obabel "{out_pdbqt}" -O "{out_sdf}" 2>/dev/null')
    scores, cur_model = [], None
    for line in open(out_pdbqt):
        ln = line.strip()
        if ln.startswith("MODEL"):
            try: cur_model = int(ln.split()[1])
            except: pass
        elif ln.startswith("REMARK VINA RESULT:"):
            try:
                p = ln.split()
                scores.append({"pose": cur_model, "affinity": float(p[3]),
                                "rmsd_lb": float(p[4]), "rmsd_ub": float(p[5])})
            except: pass
    return {
        "success": True, "out_pdbqt": out_pdbqt, "out_sdf": out_sdf,
        "scores": scores, "top_score": scores[0]["affinity"] if scores else None,
        "log": vlog,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BOND ORDER CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _bo_template(smiles: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def _bo_fix_mol(probe, template):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    probe_noH = Chem.RemoveHs(probe, sanitize=False)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, probe_noH)
    except ValueError as exc:
        raise RuntimeError(f"AssignBondOrdersFromTemplate: {exc}") from exc
    match = fixed.GetSubstructMatch(template)
    if match:
        em = Chem.RWMol(fixed)
        for ti, fi in enumerate(match):
            em.GetAtomWithIdx(fi).SetFormalCharge(
                template.GetAtomWithIdx(ti).GetFormalCharge()
            )
        fixed = em.GetMol()
    Chem.SanitizeMol(fixed)
    for prop in probe.GetPropsAsDict():
        fixed.SetProp(prop, probe.GetProp(prop))
    return fixed


def fix_sdf_bond_orders(raw_sdf: str, smiles: str, fixed_sdf: str) -> list:
    """
    Apply bond-order + formal-charge correction to all poses.
    Writes heavy-atom-only mols — never adds H without 3D coords.
    """
    import shutil
    from rdkit import Chem
    log = []
    try:
        template = _bo_template(smiles)
    except Exception as e:
        log.append(f"⚠ Could not build template: {e} — skipping fix")
        shutil.copy(raw_sdf, fixed_sdf)
        return log

    supplier = Chem.SDMolSupplier(raw_sdf, sanitize=False, removeHs=False)
    writer   = Chem.SDWriter(fixed_sdf)
    writer.SetKekulize(False)
    ok = err = 0
    for i, mol in enumerate(supplier):
        if mol is None:
            log.append(f"⚠ Pose {i+1}: unreadable — skipped"); err += 1; continue
        try:
            fixed = _bo_fix_mol(mol, template)
            writer.write(fixed); ok += 1
        except Exception as e:
            log.append(f"⚠ Pose {i+1}: fix failed ({e}) — writing raw")
            writer.write(Chem.RemoveHs(mol, sanitize=False)); err += 1
    writer.close()
    log.append(f"Bond-order fix: {ok} OK, {err} fallback")
    return log


# ══════════════════════════════════════════════════════════════════════════════
#  SDF UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_mols_from_sdf(sdf_path: str, sanitize: bool = True) -> list:
    from rdkit import Chem
    return [m for m in Chem.SDMolSupplier(sdf_path, sanitize=sanitize,
                                           removeHs=False) if m is not None]


def write_single_pose(mol, path: str) -> None:
    from rdkit import Chem
    with Chem.SDWriter(path) as w:
        w.write(mol)


def convert_sdf_to_v2000(sdf_path: str) -> str:
    out = sdf_path.replace(".sdf", "_v2000.sdf")
    if out == sdf_path:
        out = sdf_path + "_v2000.sdf"
    rc, _ = run_cmd(f'obabel "{sdf_path}" -O "{out}" --gen3d -h 2>/dev/null')
    if rc == 0 and os.path.exists(out) and os.path.getsize(out) > 10:
        return out
    rc, _ = run_cmd(f'obabel "{sdf_path}" -O "{out}" 2>/dev/null')
    if rc == 0 and os.path.exists(out) and os.path.getsize(out) > 10:
        return out
    return sdf_path


# ══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def get_interacting_residues(receptor_pdb: str, lig_mol, cutoff: float = 3.5) -> list:
    try:
        import numpy as np
        from prody import parsePDB
        conf    = lig_mol.GetConformer()
        lig_xyz = np.array([[conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z]
                             for i in range(lig_mol.GetNumAtoms())])
        rec      = parsePDB(receptor_pdb)
        r_xyz    = rec.getCoords()
        chains   = rec.getChids()
        resids   = rec.getResnums()
        resnames = rec.getResnames()
        seen = {}
        for j in range(len(r_xyz)):
            if np.linalg.norm(lig_xyz - r_xyz[j], axis=1).min() <= cutoff:
                key = (str(chains[j]), int(resids[j]))
                if key not in seen:
                    seen[key] = str(resnames[j])
        return [{"chain": k[0], "resi": k[1], "resn": v} for k, v in seen.items()]
    except Exception:
        return []


def _min_dist_residue(receptor_pdb: str, lig_mol, chain: str, resid: int) -> float:
    """Minimum heavy-atom distance between a specific residue and the ligand."""
    try:
        import numpy as np
        from prody import parsePDB
        rec = parsePDB(receptor_pdb)
        if rec is None:
            return 999.0
        res = rec.select(f"chain {chain} resnum {resid}")
        if res is None:
            return 999.0
        conf    = lig_mol.GetConformer()
        lig_xyz = np.array([[conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z]
                             for i in range(lig_mol.GetNumAtoms())])
        res_xyz = res.getCoords()
        return float(min(
            np.linalg.norm(lp - rp)
            for lp in lig_xyz for rp in res_xyz
        ))
    except Exception:
        return 999.0


def _classify_interaction_full(
    receptor_pdb: str, lig_mol, chain: str, resid: int
) -> str:
    """
    Classify residue–ligand interaction into one of eight types:
    hbond | hbond_to_halogen | pi_pi | cation_pi |
    hydrophobic | ionic | metal | halogen
    """
    try:
        import numpy as np
        from prody import parsePDB
        from rdkit import Chem

        rec = parsePDB(receptor_pdb)
        if rec is None:
            return "hydrophobic"
        res = rec.select(f"chain {chain} resnum {resid}")
        if res is None:
            return "hydrophobic"

        resnames_set = set(res.getResnames())
        resname      = list(resnames_set)[0] if resnames_set else ""

        # 1. Metal coordination
        if resname.upper() in METAL_RESNAMES:
            return "metal"

        # Compute minimum distance
        conf    = lig_mol.GetConformer()
        lig_xyz = np.array([[conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z]
                             for i in range(lig_mol.GetNumAtoms())])
        res_xyz  = res.getCoords()
        min_dist = float(min(
            np.linalg.norm(lp - rp) for lp in lig_xyz for rp in res_xyz
        ))

        # Ligand properties (cached light analysis)
        lig_noH = Chem.RemoveHs(lig_mol, sanitize=False)
        try:
            Chem.SanitizeMol(lig_noH)
        except Exception:
            pass
        has_halogen  = any(
            lig_noH.GetAtomWithIdx(i).GetAtomicNum() in (9, 17, 35, 53)
            for i in range(lig_noH.GetNumAtoms())
        )
        has_aromatic = any(
            lig_noH.GetAtomWithIdx(i).GetIsAromatic()
            for i in range(lig_noH.GetNumAtoms())
        )
        lig_charge = Chem.GetFormalCharge(lig_noH)

        # 2. Ionic interaction
        _charged_all = {"LYS", "ARG", "HIS", "ASP", "GLU"}
        if resname in _charged_all and min_dist <= 4.0 and lig_charge != 0:
            return "ionic"

        # 3. H-bond (check halogen donor/acceptor)
        _hbond_res = {
            "SER", "THR", "TYR", "ASN", "GLN", "HIS", "LYS", "ARG",
            "ASP", "GLU", "CYS", "TRP", "HOH", "WAT", "GLY", "MET",
        }
        # Also backbone N/O atoms can H-bond regardless of residue
        res_atomnames = set(res.getNames())
        backbone_hbond = bool(res_atomnames & {"N", "O", "OXT", "NH1", "NH2"})

        if (resname in _hbond_res or backbone_hbond) and min_dist <= 3.5:
            return "hbond_to_halogen" if has_halogen else "hbond"

        # 4. π-π stacking and cation-π
        _aromatic_res = {"PHE", "TYR", "TRP", "HIS"}
        _cation_res   = {"LYS", "ARG", "HIS"}
        if resname in _aromatic_res and has_aromatic and min_dist <= 5.5:
            if resname in _cation_res:
                return "cation_pi"
            return "pi_pi"
        if resname in _cation_res and has_aromatic and min_dist <= 5.0:
            return "cation_pi"

        # 5. Hydrophobic
        _hydrophob_res = {
            "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "GLY", "TYR",
        }
        if resname in _hydrophob_res:
            return "hydrophobic"

        # 6. Catch-all: if close enough treat as H-bond
        if min_dist <= 3.5:
            return "hbond_to_halogen" if has_halogen else "hbond"

        return "hydrophobic"

    except Exception:
        return "hydrophobic"


def calc_rmsd_heavy(pose_mol, crystal_pdb_path: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        import numpy as np
        if not os.path.exists(crystal_pdb_path):
            return None
        cryst = None
        for sanitize, removeHs, proxBonding in [
            (True, True, True), (False, True, True),
            (True, True, False), (False, True, False),
        ]:
            try:
                cryst = Chem.MolFromPDBFile(crystal_pdb_path, sanitize=sanitize,
                                             removeHs=removeHs,
                                             proximityBonding=proxBonding)
                if cryst is not None and cryst.GetNumConformers() > 0:
                    if not sanitize:
                        try: Chem.SanitizeMol(cryst)
                        except: pass
                    break
                cryst = None
            except: cryst = None
        if cryst is None or cryst.GetNumConformers() == 0:
            return None

        pose = Chem.RemoveHs(pose_mol, sanitize=False)
        try: Chem.SanitizeMol(pose)
        except: pass
        if pose.GetNumConformers() == 0:
            return None

        n_smaller = min(pose.GetNumAtoms(), cryst.GetNumAtoms())
        mcs = rdFMCS.FindMCS(
            [pose, cryst], timeout=10,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            completeRingsOnly=False, matchValences=False,
        )
        if mcs.numAtoms < 3 or mcs.numAtoms < 0.6 * n_smaller:
            return None
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return None
        pm = pose.GetSubstructMatches(mcs_mol,  uniquify=False)
        cm = cryst.GetSubstructMatches(mcs_mol, uniquify=False)
        if not pm or not cm:
            return None
        pc, cc = pose.GetConformer(), cryst.GetConformer()
        def _rmsd(p_match, c_match):
            sq = sum(
                (pc.GetAtomPosition(pi).x - cc.GetAtomPosition(ci).x)**2 +
                (pc.GetAtomPosition(pi).y - cc.GetAtomPosition(ci).y)**2 +
                (pc.GetAtomPosition(pi).z - cc.GetAtomPosition(ci).z)**2
                for pi, ci in zip(p_match, c_match)
            )
            return float(np.sqrt(sq / len(p_match)))
        return min(_rmsd(p, c) for p in pm for c in cm)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW REST API
# ══════════════════════════════════════════════════════════════════════════════

_PP_BASE          = "https://proteins.plus/api/v2/"
_PP_UPLOAD        = _PP_BASE + "molecule_handler/upload/"
_PP_UPLOAD_JOBS   = _PP_BASE + "molecule_handler/upload/jobs/"
_PP_POSEVIEW      = _PP_BASE + "poseview/"
_PP_POSEVIEW_JOBS = _PP_BASE + "poseview/jobs/"
_PP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://proteins.plus/",
    "Origin":  "https://proteins.plus",
}
_PP_PROTEIN_CACHE: dict = {}


def _pp_poll(job_id, poll_url, poll_interval=2, max_polls=60) -> dict:
    import requests
    job    = requests.get(poll_url + job_id + "/", headers=_PP_HEADERS, timeout=15).json()
    status = str(job.get("status", "")).lower()
    polls  = 0
    while status in ("pending", "running", "processing", "queued", ""):
        if polls >= max_polls:
            raise RuntimeError(f"Job {job_id} still '{status}' after {max_polls*poll_interval}s")
        time.sleep(poll_interval)
        polls += 1
        job    = requests.get(poll_url + job_id + "/", headers=_PP_HEADERS, timeout=15).json()
        status = str(job.get("status", "")).lower()
    return job


def call_poseview_v1(receptor_pdb: str, pose_sdf: str) -> tuple:
    import requests, io as _io
    _PROTEINS = _PP_BASE + "molecule_handler/proteins/"
    last_error = "Unknown error"
    for attempt in range(1, _PV_MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(_PV_RETRY_DELAY)
        try:
            if receptor_pdb in _PP_PROTEIN_CACHE:
                pdb_text = _PP_PROTEIN_CACHE[receptor_pdb]
            else:
                with open(receptor_pdb) as f:
                    r = requests.post(_PP_UPLOAD, files={"protein_file": f},
                                      headers=_PP_HEADERS, timeout=60)
                r.raise_for_status()
                job_id = r.json().get("job_id") or r.json().get("id")
                if not job_id:
                    last_error = f"Protein upload: no job_id in {r.json()}"
                    continue
                job = _pp_poll(job_id, _PP_UPLOAD_JOBS)
                if str(job.get("status", "")).lower() != "success":
                    last_error = f"MoleculeHandler failed (attempt {attempt}): {job}"
                    continue
                protein_id   = job["output_protein"]
                protein_json = requests.get(
                    _PROTEINS + protein_id + "/", headers=_PP_HEADERS, timeout=15
                ).json()
                pdb_text = protein_json.get("file_string", "")
                if not pdb_text:
                    last_error = f"file_string missing. Keys: {list(protein_json.keys())}"
                    continue
                _PP_PROTEIN_CACHE[receptor_pdb] = pdb_text
        except Exception as e:
            last_error = f"Protein upload (attempt {attempt}): {e}"
            continue

        try:
            ligand_v2000 = convert_sdf_to_v2000(pose_sdf)
            with open(ligand_v2000) as lf:
                r = requests.post(
                    _PP_POSEVIEW,
                    files={
                        "protein_file": ("receptor.pdb", _io.StringIO(pdb_text), "chemical/x-pdb"),
                        "ligand_file":  lf,
                    },
                    headers=_PP_HEADERS, timeout=30,
                )
            r.raise_for_status()
            pv_job_id = r.json().get("job_id") or r.json().get("id")
            if not pv_job_id:
                last_error = f"PoseView submission: no job_id in {r.json()}"
                continue
        except Exception as e:
            last_error = f"PoseView submission (attempt {attempt}): {e}"
            continue

        try:
            pv_job = _pp_poll(pv_job_id, _PP_POSEVIEW_JOBS)
            status = str(pv_job.get("status", "")).lower()
        except RuntimeError as e:
            last_error = str(e)
            continue
        except Exception as e:
            last_error = f"Polling (attempt {attempt}): {e}"
            continue

        if status in ("failed", "failure", "error"):
            last_error = f"PoseView rejected job (attempt {attempt}). Response: {pv_job}"
            continue
        if status != "success":
            last_error = f"Unexpected status '{status}' (attempt {attempt}). Response: {pv_job}"
            continue

        img_url = pv_job.get("image")
        if not img_url:
            last_error = f"Job succeeded but 'image' missing. Keys: {list(pv_job.keys())}"
            continue
        try:
            resp = requests.get(img_url, headers=_PP_HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.content, None
        except Exception as e:
            last_error = f"SVG download (attempt {attempt}): {e}"

    return None, last_error


def warm_poseview_cache(receptor_pdb: str) -> tuple:
    import requests
    _PROTEINS = _PP_BASE + "molecule_handler/proteins/"
    try:
        if receptor_pdb in _PP_PROTEIN_CACHE:
            return True, "Already cached"
        with open(receptor_pdb) as f:
            r = requests.post(_PP_UPLOAD, files={"protein_file": f},
                               headers=_PP_HEADERS, timeout=60)
        r.raise_for_status()
        job_id = r.json().get("job_id") or r.json().get("id")
        if not job_id:
            return False, f"No job_id: {r.json()}"
        job = _pp_poll(job_id, _PP_UPLOAD_JOBS)
        if str(job.get("status", "")).lower() != "success":
            return False, f"Upload failed: {job}"
        protein_id   = job["output_protein"]
        protein_json = requests.get(
            _PROTEINS + protein_id + "/", headers=_PP_HEADERS, timeout=15
        ).json()
        pdb_text = protein_json.get("file_string", "")
        if not pdb_text:
            return False, "No file_string"
        _PP_PROTEIN_CACHE[receptor_pdb] = pdb_text
        return True, f"Cached ({len(pdb_text)} chars)"
    except Exception as e:
        return False, str(e)


def clear_poseview_cache():
    _PP_PROTEIN_CACHE.clear()


def call_poseview2_ref(pdb_code: str, ligand_id: str) -> tuple:
    import requests
    _SUBMIT    = "https://proteins.plus/api/poseview2_rest"
    last_error = "Unknown error"
    for attempt in range(1, _PV_MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(_PV_RETRY_DELAY)
        try:
            r = requests.post(
                _SUBMIT,
                json={"poseview2": {"pdbCode": pdb_code.strip().lower(),
                                    "ligand": ligand_id.strip()}},
                headers={"Accept": "application/json",
                         "Content-Type": "application/json"},
                timeout=30,
            )
            data = r.json()
            if r.status_code not in (200, 202):
                last_error = f"Submission {r.status_code} (attempt {attempt}): {data}"
                continue
            location = data.get("location", "")
            if not location:
                last_error = f"No job location in: {data}"
                continue
        except Exception as e:
            last_error = f"Submission (attempt {attempt}): {e}"
            continue

        job_failed = False
        for poll_i in range(_PV_POLL_ATTEMPTS):
            time.sleep(2)
            try:
                poll = requests.get(location, headers={"Accept": "application/json"},
                                    timeout=15).json()
                sc   = poll.get("status_code")
                if sc == 200:
                    svg_url = poll.get("result_svg", "")
                    if not svg_url:
                        last_error = f"result_svg empty: {poll}"; job_failed = True; break
                    resp = requests.get(svg_url, timeout=20)
                    resp.raise_for_status()
                    return resp.content, None
                elif sc == 202:
                    continue
                else:
                    last_error = f"Poll status {sc} (attempt {attempt}): {poll}"
                    job_failed = True; break
            except Exception as e:
                last_error = f"Poll (attempt {attempt}, poll {poll_i+1}): {e}"
                continue
        if not job_failed:
            last_error = f"Timed out {_PV_POLL_ATTEMPTS*2}s (attempt {attempt})"

    return None, last_error


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def svg_to_png(svg_bytes: bytes):
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_bytes, scale=2, background_color="white")
    except Exception:
        return None


def stamp_png(png_bytes: bytes, text: str) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io
        img  = Image.open(_io.BytesIO(png_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = None
        for fp, sz in [
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 28),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 26),
        ]:
            try: font = ImageFont.truetype(fp, sz); break
            except: pass
        if font is None:
            font = ImageFont.load_default()
        bbox       = draw.textbbox((0, 0), text, font=font, anchor="lt")
        tw, th     = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = 36, 16
        pill_w, pill_h = tw + pad_x * 2, th + pad_y * 2
        pill_r = pill_h // 2
        px     = (img.width - pill_w) // 2
        py_    =  img.height - pill_h - 28
        draw.rounded_rectangle([px, py_, px + pill_w, py_ + pill_h],
                                radius=pill_r, fill=(232, 232, 232, 230))
        draw.text((px + pill_w // 2, py_ + pill_h // 2), text, font=font,
                  fill=(26, 26, 26, 255), anchor="mm")
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return png_bytes


def _svg_stamp(svg_text: str, title: str, w: int, h: int) -> str:
    _esc  = title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    pad   = int(w * 0.05)
    pw    = w - 2 * pad
    ph    = 28
    py_   = h - ph - 8
    ty_   = py_ + ph // 2
    rx    = ph // 2
    stamp = (
        f'<g>'
        f'<rect x="{pad}" y="{py_}" width="{pw}" height="{ph}"'
        f' rx="{rx}" ry="{rx}" fill="#E8E8E8" fill-opacity="0.93"'
        f' stroke="#C8C8C8" stroke-width="0.5"/>'
        f'<text x="{w//2}" y="{ty_}" text-anchor="middle"'
        f' dominant-baseline="middle"'
        f' font-family="Helvetica Neue,Arial,sans-serif"'
        f' font-size="13" font-weight="500" fill="#1A1A1A">{_esc}</text>'
        f'</g>'
    )
    return svg_text.replace("</svg>", f"{stamp}</svg>")


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_poseview() -> dict:
    import requests
    result = {"server_reachable": False, "upload_ok": False, "poseview_ok": False,
              "status": "", "job_response": {}, "image_url": "", "error": "", "log": []}
    log = result["log"]
    try:
        r = requests.get("https://proteins.plus/api/v2/", timeout=10)
        r.raise_for_status()
        result["server_reachable"] = True
        log.append(f"✓ Server reachable (HTTP {r.status_code})")
    except Exception as e:
        result["error"] = f"Server unreachable: {e}"
        log.append(f"✗ Server unreachable: {e}")
        return result
    try:
        r = requests.post(_PP_UPLOAD, data={"pdb_code": "4agn"}, timeout=30)
        r.raise_for_status()
        job_id = r.json().get("job_id")
        log.append(f"✓ Upload job: {job_id}")
        job = _pp_poll(job_id, _PP_UPLOAD_JOBS, poll_interval=1, max_polls=30)
        protein_id   = job["output_protein"]
        protein_json = requests.get(
            _PP_BASE + "molecule_handler/proteins/" + protein_id + "/", timeout=15
        ).json()
        pdb_text   = protein_json["file_string"]
        ligand_id  = protein_json["ligand_set"][0]
        ligand_json = requests.get(
            _PP_BASE + "molecule_handler/ligands/" + ligand_id + "/", timeout=15
        ).json()
        sdf_text = ligand_json["file_string"]
        log.append(f"✓ protein {len(pdb_text)} chars + ligand {ligand_json.get('name')}")
        result["upload_ok"] = True
    except Exception as e:
        result["error"] = f"MoleculeHandler failed: {e}"
        log.append(f"✗ {result['error']}")
        return result
    try:
        import io as _io
        r = requests.post(
            _PP_POSEVIEW,
            files={"protein_file": ("test.pdb", _io.StringIO(pdb_text), "chemical/x-pdb"),
                   "ligand_file":  ("test.sdf", _io.StringIO(sdf_text), "chemical/x-mdl-sdfile")},
            timeout=30,
        )
        r.raise_for_status()
        pv_job = _pp_poll(r.json().get("job_id"), _PP_POSEVIEW_JOBS,
                          poll_interval=2, max_polls=30)
        result["status"] = pv_job.get("status", "")
        result["job_response"] = pv_job
        if result["status"] == "success":
            result["image_url"] = pv_job.get("image", "")
            result["poseview_ok"] = True
            log.append(f"✓ PoseView ok — {result['image_url']}")
        else:
            result["error"] = f"PoseView '{result['status']}': {pv_job}"
            log.append(f"✗ {result['error']}")
    except Exception as e:
        result["error"] = f"PoseView step: {e}"
        log.append(f"✗ {result['error']}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTION DIAGRAM — SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_HBOND_CUTOFF      = 3.5
_HYDROPHOBIC_CUTOFF = 4.5

_COLOR_HBOND     = (0.35, 0.61, 0.84, 0.35)
_COLOR_HYDROPHOB = (0.17, 0.55, 0.34, 0.35)
_COLOR_OTHER     = (0.80, 0.37, 0.54, 0.35)

_POLAR_RES      = {"SER","THR","TYR","ASN","GLN","HIS","LYS","ARG",
                   "ASP","GLU","CYS","TRP","HOH","WAT"}
_HYDROPHOBIC_RES = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY"}


def _classify_interaction(receptor_pdb, lig_mol, chain, resid) -> str:
    """Simple 3-type classifier (kept for backward compat)."""
    try:
        import numpy as np
        from prody import parsePDB
        rec = parsePDB(receptor_pdb)
        res = rec.select(f"chain {chain} resnum {resid}")
        if res is None:
            return "other"
        conf    = lig_mol.GetConformer()
        lig_xyz = np.array([[conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z]
                             for i in range(lig_mol.GetNumAtoms())])
        res_xyz  = res.getCoords()
        min_dist = min(float(np.linalg.norm(lp-rp))
                       for lp in lig_xyz for rp in res_xyz)
        rns = set(res.getResnames())
        if any(r in _POLAR_RES for r in rns) and min_dist <= _HBOND_CUTOFF:
            return "hbond"
        if any(r in _HYDROPHOBIC_RES for r in rns):
            return "hydrophobic"
        return "other"
    except Exception:
        return "other"


def _closest_lig_atoms(lig_mol, receptor_pdb, chain, resid, max_atoms=2) -> list:
    try:
        import numpy as np
        from prody import parsePDB
        rec = parsePDB(receptor_pdb)
        res = rec.select(f"chain {chain} resnum {resid}")
        if res is None:
            return [0]
        conf    = lig_mol.GetConformer()
        res_xyz = res.getCoords()
        dists   = []
        for i in range(lig_mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            lp  = np.array([pos.x, pos.y, pos.z])
            d   = min(float(np.linalg.norm(lp - rp)) for rp in res_xyz)
            dists.append((d, i))
        dists.sort()
        return [idx for _, idx in dists[:max_atoms]]
    except Exception:
        return [0]


def _map_3d_to_2d(mol3d, mol2d):
    """
    Return {idx_3d: idx_2d} dict via substructure match.
    Falls back to atomic-number round-robin on failure.
    """
    from rdkit import Chem
    n2d = mol2d.GetNumAtoms()
    mol3d_noH = Chem.RemoveHs(mol3d, sanitize=False)
    try:
        Chem.SanitizeMol(mol3d_noH)
    except Exception:
        pass
    mapping = {}
    try:
        match = mol3d_noH.GetSubstructMatch(mol2d)
        if len(match) == n2d:
            for i2d, i3d in enumerate(match):
                mapping[i3d] = i2d
            return mapping
    except Exception:
        pass
    # Round-robin fallback
    mol2d_atoms = [(mol2d.GetAtomWithIdx(i).GetAtomicNum(), i)
                   for i in range(n2d)]
    used = {}
    for i3d in range(mol3d_noH.GetNumAtoms()):
        an         = mol3d_noH.GetAtomWithIdx(i3d).GetAtomicNum()
        candidates = [i for a, i in mol2d_atoms if a == an] or list(range(n2d))
        k          = used.get(an, 0) % len(candidates)
        mapping[i3d] = candidates[k]
        used[an]  = k + 1
    return mapping


def _get_lig_svg_and_atom_coords(mol2d, W: int, H: int,
                                  top_pad: int = 56,
                                  bottom_pad: int = 90,
                                  side_pad: int = 110) -> tuple:
    """
    Render the 2D ligand into a W×H canvas and return
    (inner_svg_content: str, atom_pixel_coords: dict {idx: (x, y)}).

    The inner SVG content is placed centered in the available area
    (excluding top_pad for title, bottom_pad for legend, side_pad margins).
    atom_pixel_coords are in the full W×H coordinate space.
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem     import rdDepictor

    rdDepictor.Compute2DCoords(mol2d)

    x0, y0 = side_pad, top_pad
    x1, y1 = W - side_pad, H - bottom_pad

    drawer = rdMolDraw2D.MolDraw2DSVG(W, H)
    opts   = drawer.drawOptions()
    opts.addAtomIndices   = False
    opts.padding          = 0.22
    opts.bondLineWidth    = 1.8

    # Constrain drawing area (available in RDKit ≥ 2022)
    try:
        from rdkit.Geometry import rdGeometry as _rg
        drawer.SetDrawBounds(_rg.Point2D(x0, y0), _rg.Point2D(x1, y1))
    except Exception:
        try:
            drawer.SetDrawBounds(
                rdMolDraw2D.Point2D(x0, y0),
                rdMolDraw2D.Point2D(x1, y1),
            )
        except Exception:
            pass

    drawer.DrawMolecule(mol2d)
    drawer.FinishDrawing()
    svg_full = drawer.GetDrawingText()

    # Strip outer <svg> wrapper and background rect to get inner content
    inner = _re.sub(r'<\?xml[^>]*\?>', '',    svg_full, flags=_re.DOTALL)
    inner = _re.sub(r'<svg\b[^>]*>',   '',    inner,    count=1)
    inner = _re.sub(r'</svg\s*>',       '',    inner)
    # Remove background rect including its optional closing tag
    inner = _re.sub(
        r'<rect\b[^>]*(fill\s*[=:]\s*["\']?#?[Ff][Ff][Ff]{4}|'
        r'fill\s*[=:]\s*["\']?white)[^>]*/?>(?:\s*</rect\s*>)?',
        '', inner, count=1, flags=_re.IGNORECASE | _re.DOTALL,
    )
    inner = inner.strip()

    # --- atom pixel coordinates via GetDrawCoords (RDKit ≥ 2022) ----------
    atom_coords: dict = {}
    for i in range(mol2d.GetNumAtoms()):
        try:
            pt = drawer.GetDrawCoords(i)
            atom_coords[i] = (float(pt.x), float(pt.y))
        except Exception:
            pass

    # --- fallback: manual scaling from conformer ---------------------------
    if len(atom_coords) < max(mol2d.GetNumAtoms() // 2, 1):
        import numpy as np
        conf = mol2d.GetConformer()
        N    = mol2d.GetNumAtoms()
        raw  = np.array([
            [conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y]
            for i in range(N)
        ])
        if N > 0:
            xmn, xmx = raw[:, 0].min(), raw[:, 0].max()
            ymn, ymx = raw[:, 1].min(), raw[:, 1].max()
            mol_w    = max(xmx - xmn, 1e-6)
            mol_h    = max(ymx - ymn, 1e-6)
            avail_w  = x1 - x0
            avail_h  = y1 - y0
            scale    = min(avail_w / mol_w, avail_h / mol_h) * 0.52
            cx_mol   = (xmn + xmx) / 2
            cy_mol   = (ymn + ymx) / 2
            cx_svg   = (x0 + x1) / 2
            cy_svg   = (y0 + y1) / 2
            for i in range(N):
                atom_coords[i] = (
                    float((raw[i, 0] - cx_mol) * scale + cx_svg),
                    float((raw[i, 1] - cy_mol) * scale + cy_svg),
                )

    return inner, atom_coords


def _place_residue_bubbles(
    atom_coords: dict,
    residue_nearest: list,   # list of (atom_idx: int, min_dist: float)
    W: int, H: int,
    top_pad: int = 56, bottom_pad: int = 90, side_pad: int = 60,
    bubble_r: float = 24.5,
) -> list:
    """
    Radial bubble placement around the ligand.

    Returns list of (lx, ly, bx, by) in the same order as residue_nearest.
    lx,ly = SVG coord of nearest ligand atom (line attachment on ligand side)
    bx,by = initial bubble centre (draggable in interactive mode)
    """
    import numpy as np

    if not atom_coords or not residue_nearest:
        return []

    xs = [v[0] for v in atom_coords.values()]
    ys = [v[1] for v in atom_coords.values()]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    lig_r   = max(np.hypot(v[0] - cx, v[1] - cy) for v in atom_coords.values())
    ring_r  = max(lig_r + bubble_r * 2.8, 130.0)

    # Ideal angle = direction from centroid toward nearest atom
    ideal_angles = []
    for atom_idx, _dist in residue_nearest:
        ax, ay = atom_coords.get(atom_idx, (cx, cy))
        ideal_angles.append(float(np.arctan2(ay - cy, ax - cx)))

    N       = len(ideal_angles)
    order   = list(np.argsort(ideal_angles))
    s_ideal = [ideal_angles[k] for k in order]

    # Enforce minimum angular separation
    min_sep = max(2 * np.pi / max(N, 1), 0.38)
    placed  = [s_ideal[0]] if s_ideal else []
    for i in range(1, len(s_ideal)):
        prev = placed[-1]
        curr = s_ideal[i]
        diff = curr - prev
        while diff >  np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        if abs(diff) < min_sep:
            curr = prev + (np.sign(diff) if diff != 0 else 1.0) * min_sep
        placed.append(curr)

    # Map back to original order
    final_angles = [0.0] * N
    for new_k, orig_k in enumerate(order):
        final_angles[orig_k] = placed[new_k] if new_k < len(placed) else ideal_angles[orig_k]

    bx_min = side_pad + bubble_r
    bx_max = W - side_pad - bubble_r
    by_min = top_pad + bubble_r
    by_max = H - bottom_pad - bubble_r

    result = []
    for i, (atom_idx, _dist) in enumerate(residue_nearest):
        lx, ly = atom_coords.get(atom_idx, (cx, cy))
        angle  = final_angles[i]
        bx     = float(np.clip(cx + ring_r * np.cos(angle), bx_min, bx_max))
        by     = float(np.clip(cy + ring_r * np.sin(angle), by_min, by_max))
        result.append((float(lx), float(ly), bx, by))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ANYONE CAN DOCK — CUSTOM 2D DIAGRAM  (Tab 1 in the UI)
# ══════════════════════════════════════════════════════════════════════════════

def draw_interaction_diagram_data(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    max_residues: int = 14,
) -> dict:
    """
    Produce the data dict consumed by _render_interactive_diagram() in app.py.

    Returns dict with keys:
        W, H, title, ligand_svg, placements
    or None on fatal error.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    import numpy as np

    W, H = _DIAG_W, _DIAG_H

    # ── Load 3D pose ─────────────────────────────────────────────────────────
    mols = load_mols_from_sdf(pose_sdf)
    if not mols:
        return None
    mol3d = mols[0]

    # ── 2D mol from SMILES ───────────────────────────────────────────────────
    mol2d = _parse_smiles_robust(smiles)
    if mol2d is None:
        mol2d = Chem.RemoveHs(mol3d, sanitize=False)
        try:
            Chem.SanitizeMol(mol2d)
        except Exception:
            pass
    rdDepictor.Compute2DCoords(mol2d)

    # ── Render ligand SVG + atom pixel coords ─────────────────────────────────
    lig_svg_inner, atom_coords = _get_lig_svg_and_atom_coords(mol2d, W, H)
    if not atom_coords:
        return None

    # ── Map 3D → 2D atom indices ─────────────────────────────────────────────
    idx3d_to_2d = _map_3d_to_2d(mol3d, mol2d)
    n2d         = mol2d.GetNumAtoms()

    # ── Get interacting residues sorted by distance ───────────────────────────
    residues = get_interacting_residues(receptor_pdb, mol3d, cutoff=cutoff)
    residues = sorted(
        residues,
        key=lambda r: _min_dist_residue(receptor_pdb, mol3d, r["chain"], r["resi"])
    )[:max_residues]

    if not residues:
        return {
            "W": W, "H": H, "title": title,
            "ligand_svg": lig_svg_inner,
            "placements": [],
        }

    # ── Build residue data (nearest atom, distance, type) ────────────────────
    res_nearest = []   # (2d_atom_idx, min_dist)
    res_meta    = []   # parallel list of full metadata

    for res in residues:
        chain = res["chain"]
        resid = res["resi"]
        resn  = res["resn"]
        itype = _classify_interaction_full(receptor_pdb, mol3d, chain, resid)
        dist  = _min_dist_residue(receptor_pdb, mol3d, chain, resid)

        close3d = _closest_lig_atoms(mol3d, receptor_pdb, chain, resid, max_atoms=1)
        idx3d   = close3d[0] if close3d else 0
        idx2d   = idx3d_to_2d.get(idx3d, 0)
        if idx2d >= n2d:
            idx2d = 0

        res_nearest.append((idx2d, dist))
        res_meta.append({
            "id":    f"res_{len(res_meta)}_{resn}{resid}{chain}",
            "itype": itype,
            "label": f"{resn}{resid}{chain}",
            "dist":  dist,
        })

    # ── Place bubbles ─────────────────────────────────────────────────────────
    positions = _place_residue_bubbles(atom_coords, res_nearest, W, H)
    if len(positions) != len(res_meta):
        return None

    # ── Build placements list ─────────────────────────────────────────────────
    # distance is shown on lines only for directional interactions
    _show_dist_types = {"hbond", "hbond_to_halogen", "ionic", "metal", "halogen"}
    placements = []
    for i, (meta, (lx, ly, bx, by)) in enumerate(zip(res_meta, positions)):
        dist_str = (
            f"{meta['dist']:.1f}"
            if meta["itype"] in _show_dist_types and meta["dist"] < 9.0
            else None
        )
        placements.append({
            "id":       meta["id"],
            "itype":    meta["itype"],
            "label":    meta["label"],
            "lx":       round(lx, 1),
            "ly":       round(ly, 1),
            "bx":       round(bx, 1),
            "by":       round(by, 1),
            "distance": dist_str,
        })

    return {
        "W":          W,
        "H":          H,
        "title":      title,
        "ligand_svg": lig_svg_inner,
        "placements": placements,
    }


def draw_interaction_diagram(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    max_residues: int = 14,
) -> bytes:
    """
    Static SVG version of the custom 2D interaction diagram.
    Uses the same data as draw_interaction_diagram_data().
    Falls back to the basic RDKit diagram on any error.
    """
    data = draw_interaction_diagram_data(
        receptor_pdb, pose_sdf, smiles, title, cutoff, max_residues
    )
    if data is None:
        # Fallback: basic RDKit ZERO-bond diagram
        try:
            mols = load_mols_from_sdf(pose_sdf)
            if mols:
                return draw_interactions_rdkit(
                    mols[0], receptor_pdb, smiles, title, cutoff,
                    (int(_DIAG_W), int(_DIAG_H)), max_residues,
                )
        except Exception:
            pass
        return b'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="200"><text x="450" y="100" text-anchor="middle" font-family="Arial" font-size="16" fill="#555">Diagram unavailable</text></svg>'

    W     = data["W"]
    H     = data["H"]
    title = data["title"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
    ]

    # Title pill
    tw = min(len(title) * 13 + 48, W - 40)
    px = (W - tw) / 2
    parts += [
        f'<rect x="{px:.0f}" y="10" width="{tw:.0f}" height="40"'
        f' rx="20" ry="20" fill="#f2f2f2"/>',
        f'<text x="{W/2:.0f}" y="34" text-anchor="middle" dominant-baseline="central"'
        f' font-family="Arial,sans-serif" font-size="17" font-weight="700"'
        f' fill="#1a1a1a">{title}</text>',
    ]

    # Lines
    for p in data["placements"]:
        cfg = _DIAG_TYPE_CFG.get(p["itype"], _DIAG_TYPE_CFG["hbond"])
        if cfg["lineclr"]:
            parts.append(
                f'<line x1="{p["lx"]}" y1="{p["ly"]}"'
                f' x2="{p["bx"]}" y2="{p["by"]}"'
                f' stroke="{cfg["lineclr"]}" stroke-width="{cfg["lw"]}"'
                f' stroke-dasharray="{cfg["dash"]}" opacity="0.85"/>'
            )
            if p["distance"]:
                mx  = (p["lx"] + p["bx"]) / 2
                my  = (p["ly"] + p["by"]) / 2
                dx  = p["bx"] - p["lx"]
                dy  = p["by"] - p["ly"]
                ln  = max((dx**2 + dy**2)**0.5, 0.001)
                ox, oy = -dy / ln * 13, dx / ln * 13
                ds  = f'{p["distance"]}\u00c5'
                dtw = len(ds) * 7 + 8
                parts += [
                    f'<rect x="{mx+ox-dtw/2:.1f}" y="{my+oy-8:.1f}"'
                    f' width="{dtw:.0f}" height="17" rx="4"'
                    f' fill="white" stroke="{cfg["lineclr"]}" stroke-width="0.5"/>',
                    f'<text x="{mx+ox:.1f}" y="{my+oy:.1f}"'
                    f' text-anchor="middle" dominant-baseline="central"'
                    f' font-family="Arial,sans-serif" font-size="11"'
                    f' font-weight="700" fill="{cfg["lineclr"]}">{ds}</text>',
                ]

    # Ligand
    parts.append(f'<g id="iac-ligand">{data["ligand_svg"]}</g>')

    # Bubbles + labels
    for p in data["placements"]:
        cfg = _DIAG_TYPE_CFG.get(p["itype"], _DIAG_TYPE_CFG["hbond"])
        parts += [
            f'<circle cx="{p["bx"]}" cy="{p["by"]}" r="24.5"'
            f' fill="{cfg["fill"]}" opacity="0.5"'
            f' stroke="{cfg["stroke"]}" stroke-width="1.5"/>',
            f'<text x="{p["bx"]}" y="{p["by"]}"'
            f' text-anchor="middle" dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="11" font-weight="700"'
            f' fill="{cfg["stroke"]}">{p["label"]}</text>',
        ]

    # Legend (bottom)
    active_types = list(dict.fromkeys(p["itype"] for p in data["placements"]))
    _LEG_LABEL = {
        "hbond": "H-bond", "hbond_to_halogen": "H···Hal",
        "pi_pi": "π-π", "cation_pi": "Cat-π",
        "hydrophobic": "Hydrophob.", "ionic": "Ionic",
        "metal": "Metal", "halogen": "Halogen",
    }
    leg_ew  = 110
    leg_tot = len(active_types) * leg_ew
    leg_x0  = (W - leg_tot) / 2
    leg_y   = H - 44
    parts.append(
        f'<rect x="{leg_x0-8:.0f}" y="{leg_y-5}" width="{leg_tot+16:.0f}" height="40"'
        f' fill="white" stroke="#e0e0e0" stroke-width="0.8" rx="6"/>'
    )
    for k, it in enumerate(active_types):
        cfg  = _DIAG_TYPE_CFG.get(it, _DIAG_TYPE_CFG["hbond"])
        ix   = leg_x0 + k * leg_ew + 14
        lbl  = _LEG_LABEL.get(it, it)
        parts.append(
            f'<circle cx="{ix:.0f}" cy="{leg_y+12}" r="7"'
            f' fill="{cfg["fill"]}" opacity="0.5"'
            f' stroke="{cfg["stroke"]}" stroke-width="1"/>'
        )
        if cfg["lineclr"]:
            parts += [
                f'<line x1="{ix+9:.0f}" y1="{leg_y+12}" x2="{ix+24:.0f}" y2="{leg_y+12}"'
                f' stroke="{cfg["lineclr"]}" stroke-width="1.6"'
                f' stroke-dasharray="{cfg["dash"]}"/>',
                f'<text x="{ix+28:.0f}" y="{leg_y+12}" dominant-baseline="central"'
                f' font-family="Arial,sans-serif" font-size="11" font-weight="700"'
                f' fill="#555">{lbl}</text>',
            ]
        else:
            parts.append(
                f'<text x="{ix+11:.0f}" y="{leg_y+12}" dominant-baseline="central"'
                f' font-family="Arial,sans-serif" font-size="11" font-weight="700"'
                f' fill="#555">{lbl}</text>'
            )

    parts.append('</svg>')
    svg_text = "\n".join(parts)
    return svg_text.encode()


# ══════════════════════════════════════════════════════════════════════════════
#  RDKIT CLASSIC HIGHLIGHT-CIRCLE DIAGRAM  (Tab 2 in the UI)
# ══════════════════════════════════════════════════════════════════════════════

def draw_interactions_rdkit_classic(
    lig_mol,
    receptor_pdb: str,
    smiles: str,
    title: str = "",
    cutoff: float = 3.5,
    size: tuple = (650, 620),
    max_residues: int = 10,
) -> bytes:
    """
    Classic RDKit highlight-circle diagram.

    Color scheme:
      Blue   (0.36, 0.61, 0.84) — H-bond / polar
      Green  (0.17, 0.55, 0.34) — Hydrophobic
      Pink   (0.80, 0.37, 0.54) — Other (ionic, π-π, etc.)

    Residue labels are injected as SVG text above each highlighted circle.
    Returns SVG bytes.
    """
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    W, H = size

    # ── 2D mol ───────────────────────────────────────────────────────────────
    mol2d = _parse_smiles_robust(smiles)
    if mol2d is None:
        mol2d = Chem.RemoveHs(lig_mol, sanitize=False)
        try:
            Chem.SanitizeMol(mol2d)
        except Exception:
            pass
    rdDepictor.Compute2DCoords(mol2d)
    n2d = mol2d.GetNumAtoms()

    # ── 3D → 2D mapping ──────────────────────────────────────────────────────
    idx3d_to_2d = _map_3d_to_2d(lig_mol, mol2d)

    # ── Interacting residues ─────────────────────────────────────────────────
    residues = get_interacting_residues(receptor_pdb, lig_mol, cutoff=cutoff)
    residues = sorted(
        residues,
        key=lambda r: _min_dist_residue(receptor_pdb, lig_mol, r["chain"], r["resi"])
    )[:max_residues]

    # ── Build per-atom highlight data ─────────────────────────────────────────
    # atom_idx → (color_rgb, [label, ...])
    atom_info: dict = {}
    highlight_atoms  = []
    highlight_colors = {}
    highlight_radii  = {}

    for res in residues:
        chain = res["chain"]
        resid = res["resi"]
        resn  = res["resn"]
        itype = _classify_interaction_full(receptor_pdb, lig_mol, chain, resid)
        color = _CLASSIC_TYPE_COLOR.get(itype, _CLASSIC_DEFAULT_CLR)
        label = f"{resn}{resid}"

        close3d = _closest_lig_atoms(lig_mol, receptor_pdb, chain, resid, max_atoms=1)
        idx3d   = close3d[0] if close3d else 0
        idx2d   = idx3d_to_2d.get(idx3d, 0)
        if idx2d >= n2d:
            idx2d = 0

        if idx2d not in atom_info:
            atom_info[idx2d] = (color, [])
        atom_info[idx2d][1].append(label)

        if idx2d not in highlight_atoms:
            highlight_atoms.append(idx2d)
        if idx2d not in highlight_colors:
            highlight_colors[idx2d] = color
        highlight_radii[idx2d] = 0.55

    # ── Draw ──────────────────────────────────────────────────────────────────
    drawer = rdMolDraw2D.MolDraw2DSVG(W, H)
    opts   = drawer.drawOptions()
    opts.circleAtoms         = True
    opts.fillHighlights      = True
    opts.continuousHighlight = False
    opts.addAtomIndices      = False
    opts.padding             = 0.15
    opts.bondLineWidth       = 1.8

    # Leave bottom strip for stamp
    try:
        from rdkit.Geometry import rdGeometry as _rg
        drawer.SetDrawBounds(_rg.Point2D(8, 8), _rg.Point2D(W - 8, H - 50))
    except Exception:
        try:
            drawer.SetDrawBounds(
                rdMolDraw2D.Point2D(8, 8),
                rdMolDraw2D.Point2D(W - 8, H - 50),
            )
        except Exception:
            pass

    drawer.DrawMolecule(
        mol2d,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    # ── Inject residue labels ─────────────────────────────────────────────────
    label_parts = []
    for idx2d, (color, labels) in atom_info.items():
        try:
            pt = drawer.GetDrawCoords(idx2d)
            ax, ay = float(pt.x), float(pt.y)
        except Exception:
            continue

        r, g, b = color
        hex_col = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for k, lbl in enumerate(labels[:2]):
            ly_ = ay - 40 - k * 14
            label_parts += [
                f'<rect x="{ax - len(lbl)*3.5 - 4:.1f}" y="{ly_ - 9:.1f}"'
                f' width="{len(lbl)*7 + 8:.0f}" height="16" rx="4"'
                f' fill="white" fill-opacity="0.88"'
                f' stroke="{hex_col}" stroke-width="0.8"/>',
                f'<text x="{ax:.1f}" y="{ly_:.1f}"'
                f' text-anchor="middle" dominant-baseline="central"'
                f' font-family="Arial,Helvetica,sans-serif"'
                f' font-size="11" font-weight="700" fill="{hex_col}">{lbl}</text>',
            ]

    if label_parts:
        svg_text = svg_text.replace(
            "</svg>",
            f'<g id="rdk-labels">{"".join(label_parts)}</g></svg>',
        )

    if title:
        svg_text = _svg_stamp(svg_text, title, W, H)

    return svg_text.encode() if isinstance(svg_text, str) else svg_text


# ══════════════════════════════════════════════════════════════════════════════
#  RDKIT ZERO-BOND DIAGRAM  (original / Greg Landrum blog approach)
# ══════════════════════════════════════════════════════════════════════════════

def draw_interactions_rdkit(
    lig_mol,
    receptor_pdb: str,
    smiles: str,
    title: str = "",
    cutoff: float = 3.5,
    size: tuple = (500, 500),
    max_residues: int = 10,
) -> bytes:
    """
    2D protein–ligand interaction diagram using ZERO-bond pseudo-atoms.
    Based on Greg Landrum's RDKit blog post (Sep 2025).
    Returns SVG bytes.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor, AllChem
    import numpy as np

    # ── 1. 2D mol ─────────────────────────────────────────────────────────────
    mol2d = _parse_smiles_robust(smiles)
    if mol2d is None:
        mol2d = Chem.RemoveHs(lig_mol, sanitize=False)
        try:
            Chem.SanitizeMol(mol2d)
        except Exception:
            pass
    rdDepictor.Compute2DCoords(mol2d)
    n2d = mol2d.GetNumAtoms()

    # ── 2. 3D → 2D mapping ────────────────────────────────────────────────────
    idx3d_to_2d = _map_3d_to_2d(lig_mol, mol2d)

    # ── 3. Interacting residues ───────────────────────────────────────────────
    residues = get_interacting_residues(receptor_pdb, lig_mol, cutoff=cutoff)
    if not residues:
        d2d = Draw.MolDraw2DSVG(size[0], size[1])
        d2d.DrawMolecule(mol2d, legend=title or "No interactions found")
        d2d.FinishDrawing()
        return d2d.GetDrawingText().encode()

    def _min_dist_r(res):
        return _min_dist_residue(receptor_pdb, lig_mol, res["chain"], res["resi"])

    residues = sorted(residues, key=_min_dist_r)[:max_residues]

    # ── 4. Interaction types & colours ────────────────────────────────────────
    interactions = []
    for res in residues:
        chain = res["chain"]
        resid = res["resi"]
        resn  = res["resn"]
        label = f"{resn} {resid}"
        itype = _classify_interaction(receptor_pdb, lig_mol, chain, resid)
        color = {"hbond":      _COLOR_HBOND,
                 "hydrophobic": _COLOR_HYDROPHOB}.get(itype, _COLOR_OTHER)

        close3d = _closest_lig_atoms(lig_mol, receptor_pdb, chain, resid, max_atoms=1)
        idx3d   = close3d[0] if close3d else 0
        idx2d   = idx3d_to_2d.get(idx3d, 0)
        if idx2d >= n2d:
            idx2d = 0
        interactions.append((label, (idx2d,), color))

    # ── 5. Blog pattern: pseudo-atoms via ZERO bonds ──────────────────────────
    lig_ext = Chem.RWMol(mol2d)
    pts, clrs = [], {}
    for aname, oaids, color in interactions:
        res_atom = Chem.Atom(0)
        res_atom.SetProp("atomLabel", aname)
        aid = lig_ext.AddAtom(res_atom)
        pts.append(aid)
        clrs[aid] = color
        for oaid in oaids:
            lig_ext.AddBond(aid, oaid, Chem.BondType.ZERO)
    rdDepictor.Compute2DCoords(lig_ext)

    # ── 6. Draw ───────────────────────────────────────────────────────────────
    w, h = size
    d2d  = Draw.MolDraw2DSVG(w, h)
    opts = d2d.drawOptions()
    opts.circleAtoms         = True
    opts.fillHighlights      = True
    opts.continuousHighlight = False
    opts.highlightRadius     = 0.5
    opts.addAtomIndices      = False
    opts.padding             = 0.15
    try:
        d2d.SetDrawBounds(0, 0, w, h - 40)
    except AttributeError:
        pass
    d2d.DrawMolecule(lig_ext, highlightAtoms=pts, highlightAtomColors=clrs)
    d2d.FinishDrawing()
    svg_text = d2d.GetDrawingText()

    if title:
        svg_text = _svg_stamp(svg_text, title, w, h)

    return svg_text.encode()
