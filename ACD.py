#!/usr/bin/env python3
"""
ACD_standalone.py — Anyone Can Dock · Standalone CLI
======================================================
Single-file docking pipeline. No Streamlit required.

Platforms : macOS Intel / Apple Silicon · Windows · Linux · Google Colab
Python    : 3.9+

Usage
-----
# Install dependencies first
python ACD_standalone.py setup

# Single ligand (SMILES)
python ACD_standalone.py dock \
    --receptor  protein.pdb \
    --smiles    "O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12" \
    --name      Baicalein \
    --ph        7.4 \
    --out       ./results

# Batch (.smi file, one "SMILES Name" per line)
python ACD_standalone.py batch \
    --receptor  protein.pdb \
    --smi       ligands.smi \
    --ph        7.4 \
    --out       ./results

# Box centering options
    --center    auto          # auto-detect co-crystal ligand (default)
    --center    "x y z"       # manual XYZ, e.g. --center "12.5 -3.1 8.0"
    --box       "16 16 16"    # box size in Å (default: 16 16 16)

# pKaNET protonation (requires pkanet_core.py alongside this file)
    --mode      pkanet
    --pubchem                 # query PubChem for experimental pKa

# Full example
python ACD_standalone.py dock \
    --receptor  1M17.pdb \
    --smiles    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC" \
    --name      Erlotinib \
    --exhaustiveness 32 \
    --out       ./erlotinib_results
"""

# ══════════════════════════════════════════════════════════════════════════════
#  STANDARD LIBRARY (always available)
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import subprocess
import tempfile
import time
import re as _re
import math as _math
import platform
import shutil
import json
import csv
import argparse
import textwrap
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_environment() -> dict:
    """
    Detect runtime environment.
    Returns dict with keys: is_colab, is_jupyter, is_terminal,
    os_name, arch, python_version
    """
    is_colab = "google.colab" in sys.modules
    if not is_colab:
        try:
            import google.colab  # noqa
            is_colab = True
        except ImportError:
            pass

    is_jupyter = False
    try:
        shell = get_ipython().__class__.__name__  # noqa
        is_jupyter = True
    except NameError:
        pass

    return {
        "is_colab":   is_colab,
        "is_jupyter": is_jupyter and not is_colab,
        "is_terminal": not (is_colab or is_jupyter),
        "os_name":    platform.system().lower(),   # linux | darwin | windows
        "arch":       platform.machine().lower(),   # x86_64 | arm64 | aarch64
        "python":     platform.python_version(),
    }


ENV = detect_environment()


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE HELPERS (no dependencies)
# ══════════════════════════════════════════════════════════════════════════════

_USE_COLOR = sys.stdout.isatty() and ENV["os_name"] != "windows"

def _c(text, code):   return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
def _green(t):        return _c(t, "32")
def _yellow(t):       return _c(t, "33")
def _red(t):          return _c(t, "31")
def _cyan(t):         return _c(t, "36")
def _bold(t):         return _c(t, "1")
def _dim(t):          return _c(t, "2")

def log(msg, level="info"):
    prefix = {
        "info":    _dim("  ·"),
        "ok":      _green("  ✓"),
        "warn":    _yellow("  ⚠"),
        "error":   _red("  ✗"),
        "section": _bold(_cyan("══")),
    }.get(level, "  ·")
    print(f"{prefix} {msg}", flush=True)

def section(title):
    bar = "═" * 60
    print(f"\n{_bold(_cyan(bar))}", flush=True)
    print(f"  {_bold(title)}", flush=True)
    print(f"{_bold(_cyan(bar))}", flush=True)

def progress_bar(current, total, width=30, label=""):
    if total == 0:
        return
    frac = min(current / total, 1.0)
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(frac * 100)
    line = f"\r  [{bar}] {pct:3d}%  {label:<30}"
    print(line, end="", flush=True)
    if current >= total:
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

_PIP_PACKAGES = [
    ("rdkit",          "rdkit",           "from rdkit import Chem"),
    ("numpy",          "numpy",           "import numpy"),
    ("prody",          "prody",           "import prody"),
    ("dimorphite_dl",  "dimorphite_dl",   "from dimorphite_dl import protonate_smiles"),
    ("meeko",          "meeko",           "from meeko import MoleculePreparation"),
    ("requests",       "requests",        "import requests"),
    ("cairosvg",       "cairosvg",        "import cairosvg"),
    ("Pillow",         "pillow",          "from PIL import Image"),
]

_CONDA_PACKAGES = {
    "rdkit":    "conda install -c conda-forge rdkit -y",
    "openbabel":"conda install -c conda-forge openbabel -y",
}

_COLAB_EXTRAS = """
# Google Colab one-liner setup
!pip install rdkit-pypi dimorphite_dl meeko prody requests pillow cairosvg
!apt-get install -y openbabel
"""


def check_dependencies(verbose=True) -> dict:
    """Check which dependencies are available. Returns {name: bool}."""
    status = {}
    for name, pip_name, import_stmt in _PIP_PACKAGES:
        try:
            exec(import_stmt)
            status[name] = True
            if verbose:
                log(f"{name}", "ok")
        except (ImportError, ModuleNotFoundError):
            status[name] = False
            if verbose:
                log(f"{name}  (missing — run: pip install {pip_name})", "warn")

    # obabel
    status["obabel"] = shutil.which("obabel") is not None
    if verbose:
        if status["obabel"]:
            log("openbabel (obabel)", "ok")
        else:
            log("openbabel  (missing — see setup instructions below)", "warn")

    return status


def install_dependencies(colab=False):
    """Auto-install missing Python dependencies via pip."""
    section("Dependency Setup")

    if colab:
        log("Google Colab detected — installing packages…", "section")
        cmds = [
            [sys.executable, "-m", "pip", "install", "-q",
             "rdkit-pypi", "dimorphite_dl", "meeko", "prody",
             "requests", "pillow", "cairosvg"],
        ]
        subprocess.run(["apt-get", "install", "-y", "-q", "openbabel"],
                       capture_output=True)
    else:
        missing = []
        for name, pip_name, import_stmt in _PIP_PACKAGES:
            try:
                exec(import_stmt)
            except (ImportError, ModuleNotFoundError):
                missing.append(pip_name)

        if not missing:
            log("All Python packages already installed.", "ok")
            return

        log(f"Installing: {', '.join(missing)}", "info")
        cmds = [[sys.executable, "-m", "pip", "install"] + missing]

    for cmd in cmds:
        rc = subprocess.run(cmd, capture_output=True).returncode
        if rc != 0:
            log(f"pip install returned code {rc} — check manually", "warn")

    # openbabel instructions
    if shutil.which("obabel") is None:
        log("openbabel not found. Install it:", "warn")
        os_name = ENV["os_name"]
        if os_name == "darwin":
            log("  brew install open-babel", "info")
        elif os_name == "linux":
            log("  sudo apt-get install openbabel  OR  conda install -c conda-forge openbabel", "info")
        elif os_name == "windows":
            log("  https://github.com/openbabel/openbabel/releases (installer)", "info")

    log("Done. Re-run your command.", "ok")


def setup_command():
    """Entry point for `python ACD_standalone.py setup`."""
    section("Anyone Can Dock — Environment Setup")
    log(f"Python {ENV['python']} · {ENV['os_name']} · {ENV['arch']}", "info")
    if ENV["is_colab"]:
        log("Environment: Google Colab", "info")
        install_dependencies(colab=True)
    else:
        install_dependencies(colab=False)
    print()
    section("Checking all dependencies")
    check_dependencies(verbose=True)
    print()
    log("Setup complete. Try: python ACD_standalone.py dock --help", "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

METAL_RESNAMES = {
    "MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K","HO",
    "LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU",
}
METAL_CHARGES = {
    "MG":2.0,"ZN":2.0,"CA":2.0,"MN":2.0,"FE":3.0,"CU":2.0,"CO":2.0,
    "NI":2.0,"CD":2.0,"HG":2.0,"HO":3.0,"LA":3.0,"CE":3.0,"PR":3.0,
    "ND":3.0,"PM":3.0,"SM":3.0,"EU":3.0,"GD":3.0,"TB":3.0,"DY":3.0,
    "ER":3.0,"TM":3.0,"YB":3.0,"LU":3.0,"NA":1.0,"K":1.0,
}
EXCLUDE_IONS = set(
    "HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG,HO,"
    "LA,CE,PR,ND,PM,SM,EU,GD,TB,DY,ER,TM,YB,LU".split(",")
)
GLYCAN_NAMES = {
    "NAG","BMA","MAN","FUC","GAL","GLC","SIA","NGA",
    "FUL","GLA","BGC","A2G","LAT","MAL","CEL","SUC",
    "TRE","GCS","NDG","NGC",
}
COFACTOR_NAMES = {
    "ATP","ADP","AMP","GTP","GDP","GMP","NAD","NAP","NDP","FAD","FMN",
    "HEM","HEC","HEA","GOL","PEG","EDO","MPD","PGE","PG4",
    "SO4","PO4","SUL","PHO","IHP","TTP","CTP","UTP","COA","SAM","SAH",
    "EPE","MES","TRS","ACT","ACY",
    "HO","LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU",
}
HEME_RESNAMES = {"HEM","HEC","HEA","HEB","HDD","HDM"}
_MIN_LIG_ATOMS = 4


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd, cwd=None, timeout=120):
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def _rdkit_six_patch():
    try:
        from rdkit import six  # noqa
    except ImportError:
        from io import StringIO as _SIO
        from types import ModuleType as _MT
        import rdkit as _rdkit
        _m = _MT("six"); _m.StringIO = _SIO; _m.PY3 = True
        _rdkit.six = _m; sys.modules["rdkit.six"] = _m


def is_cif_file(filepath: str) -> bool:
    ext = Path(filepath).suffix.lower()
    if ext in (".cif", ".mmcif"):
        return True
    try:
        with open(filepath) as f:
            if f.read(512).strip().startswith("data_"):
                return True
    except Exception:
        pass
    return False


def convert_cif_to_pdb(cif_path: str, pdb_out_path: str) -> dict:
    log_list = []
    for method, fn in [
        ("gemmi",   _cif_via_gemmi),
        ("obabel",  _cif_via_obabel),
        ("prody",   _cif_via_prody),
    ]:
        try:
            ok = fn(cif_path, pdb_out_path, log_list)
            if ok:
                return {"success": True, "pdb_path": pdb_out_path, "log": log_list}
        except Exception as e:
            log_list.append(f"⚠ {method} failed: {e}")
    return {"success": False, "pdb_path": pdb_out_path,
            "log": log_list, "error": "All CIF→PDB methods failed"}


def _cif_via_gemmi(src, dst, log_list):
    import gemmi
    doc   = gemmi.cif.read(src)
    block = doc.sole_block()
    st    = gemmi.make_structure_from_block(block)
    st.setup_entities(); st.assign_label_seq_id()
    pdb_str = st.make_pdb_headers() + st.make_pdb_string()
    Path(dst).write_text(pdb_str)
    if Path(dst).stat().st_size > 100:
        log_list.append("✓ CIF→PDB via gemmi")
        return True
    return False


def _cif_via_obabel(src, dst, log_list):
    rc, _ = run_cmd(f'obabel "{src}" -O "{dst}"')
    if rc == 0 and Path(dst).exists() and Path(dst).stat().st_size > 100:
        log_list.append("✓ CIF→PDB via OpenBabel")
        return True
    return False


def _cif_via_prody(src, dst, log_list):
    from prody import parseMMCIF, writePDB as _wPDB
    atoms = parseMMCIF(src)
    if atoms is not None and atoms.numAtoms() > 0:
        _wPDB(dst, atoms)
        if Path(dst).exists() and Path(dst).stat().st_size > 100:
            log_list.append("✓ CIF→PDB via ProDy")
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  VINA BINARY (auto-download, cross-platform)
# ══════════════════════════════════════════════════════════════════════════════

def get_vina_binary(cache_dir: str = "") -> tuple:
    """
    Download AutoDock Vina 1.2.7 for the current platform.
    Returns (path_to_vina, message).
    Tries arm64 first on Apple Silicon, falls back to x86_64 via Rosetta.
    """
    os_name = ENV["os_name"]
    arch    = ENV["arch"]

    _BASE = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/"
    _FNAMES = {
        ("linux",   "x86_64"):  "vina_1.2.7_linux_x86_64",
        ("linux",   "aarch64"): "vina_1.2.7_linux_aarch64",
        ("darwin",  "arm64"):   "vina_1.2.7_mac_aarch64",
        ("darwin",  "aarch64"): "vina_1.2.7_mac_aarch64",
        ("darwin",  "x86_64"):  "vina_1.2.7_mac_x86_64",
        ("windows", "amd64"):   "vina_1.2.7_windows_x86_64.exe",
        ("windows", "x86_64"):  "vina_1.2.7_windows_x86_64.exe",
    }

    fname = _FNAMES.get((os_name, arch))
    if fname is None:
        # generic fallback
        fname = f"vina_1.2.7_{os_name}_x86_64"
        if os_name == "windows":
            fname += ".exe"

    cache_dir = cache_dir or tempfile.gettempdir()
    dest = str(Path(cache_dir) / fname)

    if Path(dest).exists() and Path(dest).stat().st_size > 100_000:
        if os_name != "windows":
            os.chmod(dest, 0o755)
        return dest, "cached"

    import requests
    url = _BASE + fname
    log(f"Downloading Vina: {fname}", "info")
    try:
        r = requests.get(url, stream=True, timeout=180, allow_redirects=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    progress_bar(done, total, label="vina")
        if total:
            print()
    except Exception as e:
        # Apple Silicon fallback → x86_64
        if os_name == "darwin" and arch in ("arm64", "aarch64"):
            fname2 = "vina_1.2.7_mac_x86_64"
            dest2  = str(Path(cache_dir) / fname2)
            try:
                r2 = requests.get(_BASE + fname2, stream=True, timeout=180, allow_redirects=True)
                r2.raise_for_status()
                with open(dest2, "wb") as f:
                    for chunk in r2.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                dest = dest2
                log("Using x86_64 Vina via Rosetta on Apple Silicon", "warn")
            except Exception as e2:
                return None, f"Download failed: {e} / x86_64 fallback: {e2}"
        else:
            return None, f"Download failed: {e}"

    if os_name != "windows":
        os.chmod(dest, 0o755)
    return dest, f"ok ({os_name}/{arch})"


def check_obabel() -> tuple:
    if shutil.which("obabel") is None:
        return False, "obabel not found"
    _, out = run_cmd("obabel --version")
    return True, (out.splitlines()[0] if out else "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  RECEPTOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _collect_removable_ligands(atoms) -> list:
    from prody import calcCenter
    excl = EXCLUDE_IONS | GLYCAN_NAMES | COFACTOR_NAMES | HEME_RESNAMES | METAL_RESNAMES
    het  = atoms.select("hetatm and not water")
    _BB  = {"N","CA","C","O"}
    if het is None:
        return []
    results = []
    for r in het.getHierView().iterResidues():
        rn = (r.getResname() or "").strip().upper()
        if rn in excl or r.numAtoms() <= _MIN_LIG_ATOMS:
            continue
        if _BB.issubset(set(r.getNames())):
            continue
        ch  = r.getChid(); ri = r.getResnum()
        sel = (f"resname {rn} and resid {ri} and chain {ch}"
               if ch and ch.strip() else f"resname {rn} and resid {ri}")
        la = atoms.select(sel)
        if la is None or la.numAtoms() == 0:
            continue
        cx_, cy_, cz_ = (float(v) for v in calcCenter(la))
        results.append({
            "resname": rn, "chain": ch, "resid": ri, "sel_str": sel,
            "ligand_id": f"{rn}_{ch}_{ri}", "n_atoms": la.numAtoms(),
            "atoms": la, "cx": cx_, "cy": cy_, "cz": cz_,
        })
    results.sort(key=lambda d: (-d["n_atoms"], d["chain"] != "A"))
    return results


def scan_ligands(raw_pdb: str) -> list:
    try:
        from prody import parsePDB, confProDy
        confProDy(verbosity="none")
        if is_cif_file(raw_pdb):
            _tmp = tempfile.mktemp(suffix=".pdb")
            res  = convert_cif_to_pdb(raw_pdb, _tmp)
            if res["success"]:
                raw_pdb = _tmp
        atoms = parsePDB(raw_pdb)
        if atoms is None:
            return []
        return [{"resname": d["resname"], "chain": d["chain"],
                 "resid": d["resid"], "n_atoms": d["n_atoms"]}
                for d in _collect_removable_ligands(atoms)]
    except Exception:
        return []


def _strip_and_convert_receptor(rec_raw: str, wdir: Path) -> dict:
    log_list = []
    rec_fh    = str(wdir / "rec.pdb")
    rec_pdbqt = str(wdir / "rec.pdbqt")
    try:
        metal_lines = []; clean_lines = []
        with open(rec_raw) as f:
            for line in f:
                if (line[:6].strip() in ("ATOM","HETATM")
                        and line[17:20].strip().upper() in METAL_RESNAMES):
                    metal_lines.append(line)
                else:
                    clean_lines.append(line)
        rec_nometal = str(wdir / "receptor_nometal.pdb")
        Path(rec_nometal).write_text("".join(clean_lines))

        rc1, out1 = run_cmd(f'obabel "{rec_nometal}" -O "{rec_fh}" -h')
        if not Path(rec_fh).exists() or Path(rec_fh).stat().st_size < 100:
            raise ValueError(f"obabel H-addition failed (rc={rc1}): {out1[:300]}")
        log_list.append("✓ Hydrogens added")

        rc2, out2 = run_cmd(f'obabel "{rec_fh}" -O "{rec_pdbqt}" -xr --partialcharge gasteiger')
        if not Path(rec_pdbqt).exists() or Path(rec_pdbqt).stat().st_size < 100:
            raise ValueError(f"PDBQT conversion failed (rc={rc2}): {out2[:300]}")
        log_list.append("✓ PDBQT ready")

        # Re-add metals to display PDB
        if metal_lines:
            lines = [l for l in Path(rec_fh).read_text().splitlines(keepends=True)
                     if l.strip() != "END"]
            lines.extend(metal_lines); lines.append("END\n")
            Path(rec_fh).write_text("".join(lines))

        # Re-inject metals into PDBQT for scoring
        if metal_lines:
            _NO_REINJECT = {"HO","LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU"}
            pdbqt_lines = [l for l in Path(rec_pdbqt).read_text().splitlines(keepends=True)
                           if l.strip() != "END"]
            injected = 0
            for ml in metal_lines:
                try:
                    rn = ml[17:20].strip().upper()
                    if rn in _NO_REINJECT:
                        continue
                    sn = int(ml[6:11]); an = ml[12:16].strip()
                    ch = ml[21] if len(ml) > 21 else "A"; ri = int(ml[22:26])
                    x  = float(ml[30:38]); y = float(ml[38:46]); z = float(ml[46:54])
                    chg = METAL_CHARGES.get(rn, 0.0); atype = rn.capitalize()
                    pdbqt_lines.append(
                        f"HETATM{sn:5d} {an:<4s} {rn:<3s} {ch}{ri:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {chg:+.3f} {atype}\n"
                    )
                    injected += 1
                except Exception:
                    pass
            pdbqt_lines.append("END\n")
            Path(rec_pdbqt).write_text("".join(pdbqt_lines))
            if injected:
                log_list.append(f"✓ Re-injected {injected} metal atom(s)")

        return {"success": True, "rec_fh": rec_fh, "rec_pdbqt": rec_pdbqt, "log": log_list}
    except Exception as e:
        log_list.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log_list}


def write_box_pdb(filename, cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx/2, sy/2, sz/2
    corners = [(cx+dx, cy+dy, cz+dz)
               for dx in (-hx,hx) for dy in (-hy,hy) for dz in (-hz,hz)]
    with open(filename, "w") as f:
        for i, (x,y,z) in enumerate(corners, 1):
            f.write(f"HETATM{i:5d}  C   BOX A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("CONECT    1    2    3    5\nCONECT    2    1    4    6\n"
                "CONECT    3    1    4    7\nCONECT    4    2    3    8\n"
                "CONECT    5    1    6    7\nCONECT    6    2    5    8\n"
                "CONECT    7    3    5    8\nCONECT    8    4    6    7\n")


def write_vina_config(filename, cx, cy, cz, sx, sy, sz):
    Path(filename).write_text(
        f"center_x = {cx:.4f}\ncenter_y = {cy:.4f}\ncenter_z = {cz:.4f}\n"
        f"size_x = {sx}\nsize_y = {sy}\nsize_z = {sz}\n"
    )


def prepare_receptor(
    raw_pdb: str,
    wdir,
    center_mode: str = "auto",
    manual_xyz: tuple = (0.0, 0.0, 0.0),
    prody_sel: str = "",
    box_size: tuple = (16, 16, 16),
    preferred_ligand: str = "",
) -> dict:
    from prody import parsePDB, calcCenter, writePDB
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    log_list = []
    sx, sy, sz = box_size

    try:
        if is_cif_file(raw_pdb):
            log_list.append("📄 mmCIF detected — converting…")
            conv_pdb = str(wdir / "converted.pdb")
            res = convert_cif_to_pdb(raw_pdb, conv_pdb)
            log_list.extend(res["log"])
            if not res["success"]:
                raise ValueError("CIF→PDB failed")
            raw_pdb = conv_pdb

        atoms = parsePDB(raw_pdb)
        if atoms is None:
            raise ValueError("ProDy parsePDB returned None")
        log_list.append(f"✓ Parsed {atoms.numAtoms()} atoms")

        ligand_pdb_path = ""; cocrystal_ligand_id = ""
        cx = cy = cz = 0.0
        all_ligs = _collect_removable_ligands(atoms)
        primary = None

        if all_ligs:
            if preferred_ligand:
                primary = next((d for d in all_ligs
                                if d["resname"].upper() == preferred_ligand.upper()), None)
            if primary is None:
                primary = all_ligs[0]

        if primary:
            cocrystal_ligand_id = primary["ligand_id"]
            ligand_pdb_path     = str(wdir / "LIG.pdb")
            writePDB(ligand_pdb_path, primary["atoms"])
            log_list.append(f"✓ Co-crystal ligand: {primary['resname']} "
                            f"chain {primary['chain']} resid {primary['resid']}")

        if center_mode == "auto":
            if primary:
                cx, cy, cz = primary["cx"], primary["cy"], primary["cz"]
            else:
                log_list.append("⚠ No co-crystal ligand — centering on protein centroid")
                prot = atoms.select("protein")
                cx, cy, cz = (float(v) for v in calcCenter(prot or atoms))
        elif center_mode == "manual":
            cx, cy, cz = (float(v) for v in manual_xyz)
        elif center_mode == "selection":
            ref = atoms.select(prody_sel.strip())
            if ref is None or ref.numAtoms() == 0:
                raise ValueError(f"ProDy selection '{prody_sel}' matched 0 atoms")
            cx, cy, cz = (float(v) for v in calcCenter(ref))

        log_list.append(f"📍 Grid center: ({cx:.2f}, {cy:.2f}, {cz:.2f})  box: {sx}×{sy}×{sz} Å")

        # Remove ligands, strip water
        if all_ligs:
            excl_expr = " or ".join(f"({d['sel_str']})" for d in all_ligs)
            sel_str   = f"not ({excl_expr}) and not water"
        else:
            sel_str = "not water"

        rec_sel = atoms.select(sel_str)
        if rec_sel is None or rec_sel.numAtoms() == 0:
            raise ValueError("Receptor selection returned 0 atoms")

        rec_raw_path = str(wdir / "receptor_atoms.pdb")
        writePDB(rec_raw_path, rec_sel)

        # Fix blank chain IDs
        try:
            lines = Path(rec_raw_path).read_text().splitlines(keepends=True)
            coord_lines = [l for l in lines if l[:6].strip() in ("ATOM","HETATM")]
            if coord_lines and all(
                (l[21]==" " if len(l) > 21 else True) for l in coord_lines
            ):
                fixed = []
                for l in lines:
                    if l[:6].strip() in ("ATOM","HETATM") and len(l) > 21:
                        l = l[:21] + "A" + l[22:]
                    fixed.append(l)
                Path(rec_raw_path).write_text("".join(fixed))
                log_list.append("✓ Assigned chain A to blank-chain atoms")
        except Exception:
            pass

        conv = _strip_and_convert_receptor(rec_raw_path, wdir)
        log_list.extend(conv["log"])
        if not conv["success"]:
            raise ValueError(conv["error"])

        # Re-inject heme into PDBQT
        heme_lines = [l for l in Path(raw_pdb).read_text().splitlines(keepends=True)
                      if l[:6].strip() in ("ATOM","HETATM")
                      and l[17:20].strip().upper() in HEME_RESNAMES]
        if heme_lines:
            _AD4_TYPE = {"FE":"Fe","N":"NA","O":"OA","C":"A","S":"SA"}
            _AD4_CHG  = {"FE":2.0,"N":-0.4,"C":0.1,"O":-0.4,"S":0.0}
            pdbqt_lines = [l for l in Path(conv["rec_pdbqt"]).read_text().splitlines(keepends=True)
                           if l.strip() != "END"]
            injected = 0
            for hl in heme_lines:
                try:
                    sn  = int(hl[6:11]); an = hl[12:16].strip()
                    rn  = hl[17:20].strip().upper()
                    ch  = hl[21] if len(hl) > 21 else "A"; ri = int(hl[22:26])
                    x   = float(hl[30:38]); y = float(hl[38:46]); z = float(hl[46:54])
                    el  = (hl[76:78].strip().upper() if len(hl) > 76 and hl[76:78].strip()
                           else an[:1].upper())
                    chg = _AD4_CHG.get(el, 0.0); atype = f"{_AD4_TYPE.get(el,'C'):>2s}"
                    pdbqt_lines.append(
                        f"HETATM{sn:5d} {an:<4s} {rn:<3s} {ch}{ri:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {chg:+.3f} {atype}\n"
                    )
                    injected += 1
                    # also add to display pdb
                    with open(conv["rec_fh"], "a") as rf:
                        rf.write(hl)
                except Exception:
                    pass
            pdbqt_lines.append("END\n")
            Path(conv["rec_pdbqt"]).write_text("".join(pdbqt_lines))
            log_list.append(f"✓ Re-injected {injected} heme atom(s)")

        box_pdb  = str(wdir / "rec.box.pdb")
        cfg_path = str(wdir / "rec.box.txt")
        write_box_pdb(box_pdb, cx, cy, cz, sx, sy, sz)
        write_vina_config(cfg_path, cx, cy, cz, sx, sy, sz)
        log_list.append("✓ Box + config written")

        return {
            "success": True,
            "rec_fh":  conv["rec_fh"], "rec_pdbqt": conv["rec_pdbqt"],
            "box_pdb": box_pdb, "config_txt": cfg_path,
            "cx": cx, "cy": cy, "cz": cz, "sx": sx, "sy": sy, "sz": sz,
            "ligand_pdb_path": ligand_pdb_path,
            "cocrystal_ligand_id": cocrystal_ligand_id,
            "all_ligands": [{"resname": d["resname"], "chain": d["chain"],
                             "resid": d["resid"]} for d in all_ligs],
            "log": log_list,
        }
    except Exception as e:
        log_list.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log_list}


# ══════════════════════════════════════════════════════════════════════════════
#  LIGAND PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _meeko_to_pdbqt(mol, out_path: str):
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    try:
        from meeko import PDBQTWriterLegacy
        setups = prep.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(setups[0])
    except (ImportError, AttributeError):
        prep.prepare(mol)
        pdbqt_str = prep.write_pdbqt_string()
    Path(out_path).write_text(pdbqt_str)


def _ligand_charge_summary(smiles: str) -> dict:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")
    net = n_pos = n_neg = 0
    rows = []
    for a in mol.GetAtoms():
        fc = int(a.GetFormalCharge()); net += fc
        if fc > 0: n_pos += 1
        elif fc < 0: n_neg += 1
        if fc != 0:
            rows.append({"atom_idx": a.GetIdx(), "symbol": a.GetSymbol(), "formal_charge": fc})
    return {"net_charge": int(net), "charged_atoms": rows,
            "is_zwitterion": bool(n_pos > 0 and n_neg > 0 and net == 0)}


def prepare_ligand(
    smiles: str,
    name: str,
    ph: float,
    wdir,
    mode: str = "dimorphite",
    use_pubchem: bool = False,
    max_tautomers: int = 8,
    ph_window: float = 1.0,
) -> dict:
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    wdir = Path(wdir); wdir.mkdir(parents=True, exist_ok=True)
    log_list  = []
    out_pdbqt = str(wdir / f"{name}.pdbqt")
    out_sdf   = str(wdir / f"{name}_3d.sdf")

    try:
        raw  = smiles.strip()
        prot = raw
        actual_mode = mode or "dimorphite"

        if actual_mode == "neutral":
            mol_check = Chem.MolFromSmiles(raw)
            if mol_check is None:
                raise ValueError(f"Cannot parse SMILES: {raw[:60]}")
            prot = Chem.MolToSmiles(mol_check, isomericSmiles=True, canonical=True)
            log_list.append("✓ Neutral mode — charge kept as-is")

        elif actual_mode == "pkanet":
            prot, charge_pka, pka_log = _pkanet_protonate(raw, ph, use_pubchem,
                                                           max_tautomers, ph_window)
            log_list.extend(pka_log)
            actual_mode = "pkanet"

        else:  # dimorphite (default)
            try:
                from dimorphite_dl import protonate_smiles
                vs = protonate_smiles(raw, ph_min=ph, ph_max=ph, max_variants=1)
                if vs:
                    prot = vs[0] if isinstance(vs, list) else vs
                    log_list.append(f"✓ Dimorphite-DL pH {ph:.1f}")
            except Exception as e:
                log_list.append(f"⚠ Dimorphite-DL: {e}")

        mol = Chem.MolFromSmiles(prot)
        if mol is None:
            raise ValueError(f"Cannot parse protonated SMILES: {prot[:60]}")

        ci = _ligand_charge_summary(prot)
        log_list.append(f"✓ Net charge: {ci['net_charge']:+d}")

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
        log_list.append("✓ 3D conformer generated + minimised")

        with Chem.SDWriter(out_sdf) as w:
            w.write(mol)

        try:
            _meeko_to_pdbqt(mol, out_pdbqt)
            log_list.append("✓ PDBQT written (Meeko)")
        except Exception as e_meeko:
            log_list.append(f"⚠ Meeko: {e_meeko} — trying OpenBabel")
            subprocess.run(f'obabel "{out_sdf}" -O "{out_pdbqt}" -xh 2>/dev/null',
                           shell=True, timeout=30)
            if not Path(out_pdbqt).exists() or Path(out_pdbqt).stat().st_size < 10:
                raise ValueError(f"Both Meeko and OpenBabel failed: {e_meeko}")
            log_list.append("✓ PDBQT written (OpenBabel fallback)")

        return {
            "success": True, "pdbqt": out_pdbqt, "sdf": out_sdf,
            "input_smiles": raw, "prot_smiles": prot,
            "charge":      ci["net_charge"], "net_charge": ci["net_charge"],
            "charged_atoms": ci["charged_atoms"],
            "is_zwitterion": ci["is_zwitterion"],
            "protonation_mode": actual_mode,
            "log": log_list,
        }
    except Exception as e:
        log_list.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log_list}


def prepare_ligand_from_file(file_path: str, name: str, wdir) -> dict:
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    wdir = Path(wdir); wdir.mkdir(parents=True, exist_ok=True)
    log_list  = []
    out_pdbqt = str(wdir / f"{name}.pdbqt")
    out_sdf   = str(wdir / f"{name}_3d.sdf")
    ext = Path(file_path).suffix.lower()

    try:
        mol = None
        if ext == ".sdf":
            for san in (True, False):
                sup  = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=san)
                mols = [m for m in sup if m]
                if mols:
                    mol = mols[0]; break
        else:
            ob_sdf = str(wdir / f"{name}_ob.sdf")
            subprocess.run(f'obabel "{file_path}" -O "{ob_sdf}" 2>/dev/null',
                           shell=True, timeout=30)
            if Path(ob_sdf).exists() and Path(ob_sdf).stat().st_size > 10:
                sup  = Chem.SDMolSupplier(ob_sdf, removeHs=False, sanitize=True)
                mols = [m for m in sup if m]
                if mols:
                    mol = mols[0]
            if mol is None and ext == ".mol2":
                mol = Chem.MolFromMol2File(file_path, removeHs=False, sanitize=True)
            if mol is None and ext == ".pdb":
                mol = Chem.MolFromPDBFile(file_path, removeHs=False, sanitize=True)

        if mol is None:
            raise ValueError(f"Cannot read molecule from {Path(file_path).name}")

        try: Chem.SanitizeMol(mol)
        except Exception: pass

        mol = Chem.AddHs(mol, addCoords=True)
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3(); params.randomSeed = 42
            AllChem.EmbedMolecule(mol, params)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        with Chem.SDWriter(out_sdf) as w: w.write(mol)

        try:
            _meeko_to_pdbqt(mol, out_pdbqt)
        except Exception as em:
            subprocess.run(f'obabel "{out_sdf}" -O "{out_pdbqt}" -xh 2>/dev/null',
                           shell=True, timeout=30)

        try: smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception: smi = name
        charge = int(Chem.GetFormalCharge(mol))

        return {"success": True, "pdbqt": out_pdbqt, "sdf": out_sdf,
                "prot_smiles": smi, "charge": charge, "log": log_list}
    except Exception as e:
        log_list.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log_list}


def smiles_from_file(file_path: str, wdir) -> str:
    wdir = Path(wdir)
    ext  = Path(file_path).suffix.lower()
    if ext == ".sdf":
        from rdkit import Chem
        mols = [m for m in Chem.SDMolSupplier(file_path, sanitize=True) if m]
        if not mols:
            raise ValueError("No valid molecule in SDF")
        return Chem.MolToSmiles(mols[0])
    tmp = str(wdir / "lig_tmp.smi")
    run_cmd(f'obabel "{file_path}" -O "{tmp}" --canonical 2>/dev/null')
    for line in Path(tmp).read_text().splitlines():
        pts = line.strip().split(None, 1)
        if pts:
            return pts[0]
    raise ValueError("Cannot convert file to SMILES")


# ══════════════════════════════════════════════════════════════════════════════
#  pKaNET CLOUD PROTONATION (inline, no external pkanet_core required)
# ══════════════════════════════════════════════════════════════════════════════
# This is a self-contained version of the pKaNET algorithm.
# For pkanet_core.py (full version), place it alongside this file.

_PKANET_CACHE: dict = {}

_IONIZABLE_SITE_DEF = [
    ("sulfonic_acid",      "[SX4](=O)(=O)[OX2H1]",                1.0,  "acid"),
    ("carboxylic_acid",    "[CX3](=O)[OX2H1]",                    4.5,  "acid"),
    ("tetrazole",          "c1nn[nH]n1",                           4.9,  "acid"),
    ("imidazole_acid",     "c1cn[nH]c1",                           6.0,  "acid"),
    ("phosphonate",        "[PX4](=O)([OX2H1])[OX2H1,OX1-]",     6.5,  "acid"),
    ("sulfonamide_NH",     "[SX4](=O)(=O)[NX3;H1]",              10.1,  "acid"),
    ("phenol",             "c[OX2H1]",                            10.0,  "acid"),
    ("thiol_aliph",        "[CX4][SX2H1]",                        10.5,  "acid"),
    ("aniline",            "c[NX3;H1,H2;!$(N~[!#6])]",            4.6,  "base"),
    ("pyridine_like",      "[$([nX2]1:[c,n]:c:[c,n]:c1)]",        5.2,  "base"),
    ("piperazine_NH",      "[NX3;H1;R;$(N1CCNCC1)]",              8.1,  "base"),
    ("aliphatic_amine",    "[NX3;H1,H2;!$(NC=O);!$(N~[!#6;!H]);!$([nH]);!$(Nc)]", 9.5, "base"),
    ("aliphatic_amine_t",  "[NX3;H0;!$(NC=O);!$(Nc);!$([nH]);!$([N]~[!#6])]",     9.0, "base"),
    ("guanidine",          "[NX3][CX3](=[NX2])[NX3]",            13.0,  "base"),
]


_ION_SITES = None  # lazy — compiled on first use

def _get_ion_sites():
    global _ION_SITES
    if _ION_SITES is None:
        from rdkit import Chem
        _ION_SITES = []
        for lbl, sma, pka, stype in _IONIZABLE_SITE_DEF:
            pat = Chem.MolFromSmarts(sma)
            if pat:
                _ION_SITES.append((lbl, pat, pka, stype))
    return _ION_SITES


def _hh_fraction(pka, ph, stype):
    if stype == "acid":
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def _pkanet_protonate(smiles, ph, use_pubchem=False,
                      max_tautomers=8, ph_window=1.0) -> tuple:
    """
    Inline pKaNET protonation. Tries pkanet_core.py first (full version),
    then falls back to this built-in implementation.
    Returns (best_smiles, charge, log_list).
    """
    from rdkit import Chem
    log_list = []

    # Try external pkanet_core.py
    try:
        import pkanet_core as _pk
        _tmp = tempfile.mkdtemp(prefix="pkanet_")
        result = _pk.run_job(
            input_type="SMILES", smiles_text=smiles,
            uploaded_bytes=None, uploaded_name=None,
            target_pH=ph, output_name="lig", out_dir=_tmp,
            output_formats=[], enumerate_stereoisomers=False,
            use_pubchem=use_pubchem, ph_window=ph_window,
            max_tautomers=max_tautomers, top_n_microstates=1,
            write_alt_3d_for_top_k=0,
        )
        ligs = result.get("ligands", [])
        if ligs and ligs[0].get("selected_microstate_smiles"):
            best = ligs[0]["selected_microstate_smiles"]
            mol  = Chem.MolFromSmiles(best)
            chg  = int(Chem.GetFormalCharge(mol)) if mol else 0
            log_list.append(f"✓ pKaNET Cloud (external) — charge {chg:+d}")
            return best, chg, log_list
    except ImportError:
        log_list.append("ℹ pkanet_core.py not found — using built-in pKa heuristic")
    except Exception as e:
        log_list.append(f"⚠ pkanet_core failed ({e}) — built-in fallback")

    # Built-in: Dimorphite + ionizable site correction
    canon = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(canon)
        if mol:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            mol   = rdMolStandardize.LargestFragmentChooser().choose(mol)
            canon = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass

    prot = canon
    try:
        from dimorphite_dl import protonate_smiles
        ph_lo = max(0.0, ph - ph_window / 2)
        ph_hi = min(14.0, ph + ph_window / 2)
        vs = protonate_smiles(canon, ph_min=ph_lo, ph_max=ph_hi, max_variants=1)
        if vs:
            prot = vs[0] if isinstance(vs, list) else vs
    except Exception as e:
        log_list.append(f"⚠ Dimorphite: {e}")

    # Post-correction: deprotonate any missed acidic site
    prot = _ion_site_correction(canon, prot, ph, log_list)

    mol_out = Chem.MolFromSmiles(prot)
    charge  = int(Chem.GetFormalCharge(mol_out)) if mol_out else 0
    log_list.append(f"✓ pKaNET built-in — charge {charge:+d}")
    return prot, charge, log_list


def _ion_site_correction(original_smiles, current_smiles, ph, log_list):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(current_smiles)
    if mol is None:
        return current_smiles
    fc_map = {a.GetIdx(): int(a.GetFormalCharge()) for a in mol.GetAtoms()}
    missed = []
    for lbl, pat, pka, stype in _get_ion_sites():
        if stype != "acid" or pka >= ph:
            continue
        for match in mol.GetSubstructMatches(pat):
            for idx in match:
                a = mol.GetAtomWithIdx(idx)
                if (a.GetAtomicNum() in (7,8,16) and a.GetTotalNumHs() > 0
                        and fc_map.get(idx, 0) >= 0):
                    missed.append((pka, idx, lbl))
    if not missed:
        return current_smiles
    missed.sort()
    pka_val, target_idx, lbl = missed[0]
    try:
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(target_idx).SetFormalCharge(-1)
        Chem.SanitizeMol(rw)
        corrected = Chem.MolToSmiles(rw, canonical=True)
        log_list.append(f"✓ pKa correction: {lbl} (pKa={pka_val}) deprotonated")
        return corrected
    except Exception as e:
        log_list.append(f"⚠ pKa correction failed: {e}")
        return current_smiles


# ══════════════════════════════════════════════════════════════════════════════
#  DOCKING
# ══════════════════════════════════════════════════════════════════════════════

def run_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    config_txt: str,
    vina_path: str,
    exhaustiveness: int = 16,
    n_modes: int = 10,
    energy_range: int = 3,
    wdir = ".",
    out_name: str = "out",
) -> dict:
    wdir      = Path(wdir)
    out_pdbqt = str(wdir / f"{out_name}_out.pdbqt")
    out_sdf   = str(wdir / f"{out_name}_out.sdf")

    cmd = (f'"{vina_path}" '
           f'--receptor "{receptor_pdbqt}" '
           f'--ligand "{ligand_pdbqt}" '
           f'--config "{config_txt}" '
           f'--exhaustiveness {exhaustiveness} '
           f'--num_modes {n_modes} '
           f'--energy_range {energy_range} '
           f'--out "{out_pdbqt}"')

    rc, vlog = run_cmd(cmd, cwd=str(wdir), timeout=3600)
    if rc != 0 or not Path(out_pdbqt).exists():
        return {"success": False, "error": f"Vina exit {rc}", "log": vlog}

    run_cmd(f'obabel "{out_pdbqt}" -O "{out_sdf}" 2>/dev/null')

    scores = []; cur_model = None
    for line in Path(out_pdbqt).read_text().splitlines():
        ln = line.strip()
        if ln.startswith("MODEL"):
            try: cur_model = int(ln.split()[1])
            except Exception: pass
        elif ln.startswith("REMARK VINA RESULT:"):
            try:
                p = ln.split()
                scores.append({"pose": cur_model, "affinity": float(p[3]),
                               "rmsd_lb": float(p[4]), "rmsd_ub": float(p[5])})
            except Exception:
                pass

    return {
        "success":   True,
        "out_pdbqt": out_pdbqt,
        "out_sdf":   out_sdf,
        "scores":    scores,
        "top_score": scores[0]["affinity"] if scores else None,
        "log":       vlog,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BOND ORDER CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _bo_template(smiles: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            Chem.SetAromaticity(mol)
        except Exception:
            pass
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass  # charged/tautomeric aromatics — AssignBondOrdersFromTemplate still works
    return mol


def _bo_fix_mol(probe, template):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    probe_noH = Chem.RemoveHs(probe, sanitize=False)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, probe_noH)
    except ValueError as exc:
        raise RuntimeError(f"AssignBondOrdersFromTemplate failed: {exc}") from exc
    match = fixed.GetSubstructMatch(template)
    if match:
        em = Chem.RWMol(fixed)
        for ti, fi in enumerate(match):
            em.GetAtomWithIdx(fi).SetFormalCharge(
                template.GetAtomWithIdx(ti).GetFormalCharge())
        fixed = em.GetMol()
    Chem.SanitizeMol(fixed)
    for prop in probe.GetPropsAsDict():
        fixed.SetProp(prop, probe.GetProp(prop))
    return fixed


def fix_sdf_bond_orders(raw_sdf: str, smiles: str, fixed_sdf: str) -> list:
    import shutil
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    log_list = []
    try:
        template = _bo_template(smiles)
    except Exception as e:
        log_list.append(f"⚠ template error: {e} — skipping fix")
        shutil.copy(raw_sdf, fixed_sdf)
        RDLogger.EnableLog("rdApp.error")
        return log_list

    supplier = Chem.SDMolSupplier(raw_sdf, sanitize=False, removeHs=False)
    writer   = Chem.SDWriter(fixed_sdf)
    writer.SetKekulize(False)
    ok = err = 0
    for i, mol in enumerate(supplier):
        if mol is None:
            log_list.append(f"⚠ Pose {i+1}: unreadable — skipped"); err += 1; continue
        try:
            fixed = _bo_fix_mol(mol, template)
            try:
                fh   = Chem.AddHs(fixed, addCoords=True)
                conf = fh.GetConformer()
                bad  = any(
                    abs(conf.GetAtomPosition(j).x) + abs(conf.GetAtomPosition(j).y)
                    + abs(conf.GetAtomPosition(j).z) < 0.01
                    for j in range(fh.GetNumAtoms())
                    if fh.GetAtomWithIdx(j).GetAtomicNum() == 1
                )
                writer.write(fixed if bad else fh)
            except Exception:
                writer.write(fixed)
            ok += 1
        except Exception as e:
            log_list.append(f"⚠ Pose {i+1}: {e}")
            writer.write(Chem.RemoveHs(mol, sanitize=False)); err += 1
    writer.close()
    RDLogger.EnableLog("rdApp.error")
    log_list.append(f"Bond-order fix: {ok} OK, {err} fallback")
    return log_list


# ══════════════════════════════════════════════════════════════════════════════
#  SDF / MOL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_mols_from_sdf(sdf_path: str, sanitize: bool = True) -> list:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mols = []
    try:
        sup = Chem.SDMolSupplier(sdf_path, sanitize=sanitize, removeHs=False)
        mols = [m for m in sup if m is not None]
    except Exception:
        pass
    if sanitize and not mols:
        try:
            sup2 = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
            for m in sup2:
                if m is None: continue
                try: Chem.SanitizeMol(m)
                except Exception: pass
                mols.append(m)
        except Exception:
            pass
    RDLogger.EnableLog("rdApp.error")
    return mols


def write_single_pose(mol, path: str):
    from rdkit import Chem
    with Chem.SDWriter(path) as w: w.write(mol)


def write_single_pose_pdb(mol, path: str):
    from rdkit import Chem
    Chem.MolToPDBFile(mol, path)


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
        rec = parsePDB(receptor_pdb)
        r_xyz = rec.getCoords(); chains = rec.getChids()
        resids = rec.getResnums(); resnames = rec.getResnames()
        seen = {}
        for j in range(len(r_xyz)):
            if np.linalg.norm(lig_xyz - r_xyz[j], axis=1).min() <= cutoff:
                key = (str(chains[j]), int(resids[j]))
                if key not in seen: seen[key] = str(resnames[j])
        return [{"chain": k[0], "resi": k[1], "resn": v} for k, v in seen.items()]
    except Exception:
        return []


def calc_rmsd_heavy(pose_mol, crystal_pdb_path: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        import numpy as np
        if not Path(crystal_pdb_path).exists(): return None
        cryst = None
        for san, rh, pb in [(True,True,True),(False,True,True),(True,True,False)]:
            try:
                cryst = Chem.MolFromPDBFile(crystal_pdb_path, sanitize=san,
                                             removeHs=rh, proximityBonding=pb)
                if cryst and cryst.GetNumConformers() > 0:
                    if not san:
                        try: Chem.SanitizeMol(cryst)
                        except Exception: pass
                    break
                cryst = None
            except Exception: cryst = None
        if not cryst: return None
        pose = Chem.RemoveHs(pose_mol, sanitize=False)
        try: Chem.SanitizeMol(pose)
        except Exception: pass
        if pose.GetNumConformers() == 0: return None
        mcs = rdFMCS.FindMCS([pose, cryst], timeout=10,
                             bondCompare=rdFMCS.BondCompare.CompareAny,
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             completeRingsOnly=False, matchValences=False)
        if mcs.numAtoms < 3: return None
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if not mcs_mol: return None
        pm = pose.GetSubstructMatches(mcs_mol,  uniquify=False)
        cm = cryst.GetSubstructMatches(mcs_mol, uniquify=False)
        if not pm or not cm: return None
        pc, cc = pose.GetConformer(), cryst.GetConformer()
        def _rmsd(pm_, cm_):
            sq = sum(
                (pc.GetAtomPosition(pi).x - cc.GetAtomPosition(ci).x)**2 +
                (pc.GetAtomPosition(pi).y - cc.GetAtomPosition(ci).y)**2 +
                (pc.GetAtomPosition(pi).z - cc.GetAtomPosition(ci).z)**2
                for pi, ci in zip(pm_, cm_)
            )
            return float(np.sqrt(sq / len(pm_)))
        return min(_rmsd(p, c) for p in pm for c in cm)
    except Exception: return None


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE DOCKING WORKFLOWS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_center(center_str: str) -> tuple:
    """Parse center argument: 'auto', 'x y z', or 'resname'."""
    if not center_str or center_str.strip().lower() == "auto":
        return ("auto", (0.0, 0.0, 0.0), "")
    parts = center_str.strip().split()
    if len(parts) == 3:
        try:
            return ("manual", (float(parts[0]), float(parts[1]), float(parts[2])), "")
        except ValueError:
            pass
    # ProDy selection string
    return ("selection", (0.0, 0.0, 0.0), center_str.strip())


def _parse_box(box_str: str) -> tuple:
    """Parse box argument: 'sx sy sz' or single value for cube."""
    if not box_str:
        return (16, 16, 16)
    parts = box_str.strip().split()
    if len(parts) == 1:
        s = int(float(parts[0]))
        return (s, s, s)
    if len(parts) == 3:
        return (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
    return (16, 16, 16)


def dock_single(
    receptor:         str,
    smiles:           str,
    name:             str,
    out_dir:          str,
    ph:               float = 7.4,
    mode:             str   = "dimorphite",
    center:           str   = "auto",
    box:              str   = "16 16 16",
    exhaustiveness:   int   = 16,
    n_poses:          int   = 10,
    energy_range:     int   = 3,
    use_pubchem:      bool  = False,
    vina_path:        str   = "",
    preferred_ligand: str   = "",
) -> dict:
    """
    Complete single-ligand docking pipeline.
    Returns result dict with keys: success, scores, out_sdf, out_pdbqt,
    prot_smiles, charge, log.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Vina binary ────────────────────────────────────────────────────────
    if not vina_path:
        vina_path, vmsg = get_vina_binary(str(out_dir.parent))
        if not vina_path:
            return {"success": False, "error": f"Vina unavailable: {vmsg}"}
    log(f"Vina: {vina_path}", "ok")

    # ── 2. Receptor ───────────────────────────────────────────────────────────
    center_mode, manual_xyz, prody_sel = _parse_center(center)
    box_size = _parse_box(box)

    log(f"Preparing receptor…", "info")
    rec_result = prepare_receptor(
        raw_pdb          = receptor,
        wdir             = out_dir / "receptor",
        center_mode      = center_mode,
        manual_xyz       = manual_xyz,
        prody_sel        = prody_sel,
        box_size         = box_size,
        preferred_ligand = preferred_ligand,
    )
    if not rec_result["success"]:
        return {"success": False, "error": f"Receptor prep failed: {rec_result['error']}",
                "log": rec_result["log"]}
    for msg in rec_result["log"]: log(msg, "ok" if msg.startswith("✓") else "info")

    # ── 3. Ligand ─────────────────────────────────────────────────────────────
    log(f"Preparing ligand ({mode}, pH {ph})…", "info")
    lig_result = prepare_ligand(
        smiles=smiles, name=name, ph=ph,
        wdir=out_dir / "ligand", mode=mode,
        use_pubchem=use_pubchem,
    )
    if not lig_result["success"]:
        return {"success": False, "error": f"Ligand prep failed: {lig_result['error']}",
                "log": lig_result["log"]}
    for msg in lig_result["log"]: log(msg, "ok" if msg.startswith("✓") else "info")

    # ── 4. Docking ────────────────────────────────────────────────────────────
    log(f"Running Vina (exhaustiveness={exhaustiveness})…", "info")
    dock_result = run_vina(
        receptor_pdbqt = rec_result["rec_pdbqt"],
        ligand_pdbqt   = lig_result["pdbqt"],
        config_txt     = rec_result["config_txt"],
        vina_path      = vina_path,
        exhaustiveness = exhaustiveness,
        n_modes        = n_poses,
        energy_range   = energy_range,
        wdir           = out_dir / "docking",
        out_name       = name,
    )
    if not dock_result["success"]:
        return {"success": False, "error": dock_result["error"],
                "log": dock_result["log"]}

    # ── 5. Bond-order fix ─────────────────────────────────────────────────────
    pv_sdf = str(out_dir / f"{name}_poses.sdf")
    fix_log = fix_sdf_bond_orders(
        dock_result["out_sdf"], lig_result["prot_smiles"], pv_sdf
    )
    if not Path(pv_sdf).exists() or Path(pv_sdf).stat().st_size < 10:
        pv_sdf = dock_result["out_sdf"]

    top = dock_result["top_score"]
    if top is not None:
        cls = ("Very strong" if top < -11 else "Strong" if top < -9
               else "Moderate" if top < -7 else "Weak")
        log(f"Best pose: {top:.2f} kcal/mol  ({cls})", "ok")

    # Print score table
    if dock_result["scores"]:
        print(f"\n  {'Pose':>4}  {'Affinity (kcal/mol)':>20}")
        print(f"  {'----':>4}  {'-------------------':>20}")
        for s in dock_result["scores"]:
            print(f"  {s['pose']:>4}  {s['affinity']:>20.2f}")
        print()

    return {
        "success":     True,
        "name":        name,
        "smiles":      smiles,
        "prot_smiles": lig_result["prot_smiles"],
        "charge":      lig_result["net_charge"],
        "scores":      dock_result["scores"],
        "top_score":   top,
        "out_sdf":     pv_sdf,
        "out_pdbqt":   dock_result["out_pdbqt"],
        "rec_fh":      rec_result["rec_fh"],
        "rec_pdbqt":   rec_result["rec_pdbqt"],
        "config_txt":  rec_result["config_txt"],
        "cx": rec_result["cx"], "cy": rec_result["cy"], "cz": rec_result["cz"],
        "ligand_pdb_path": rec_result["ligand_pdb_path"],
        "cocrystal_ligand_id": rec_result["cocrystal_ligand_id"],
        "log": rec_result["log"] + lig_result["log"] + dock_result["log"] + fix_log,
    }


def dock_batch(
    receptor:        str,
    smiles_list:     list,           # [(smiles, name), ...]
    out_dir:         str,
    ph:              float = 7.4,
    mode:            str   = "dimorphite",
    center:          str   = "auto",
    box:             str   = "16 16 16",
    exhaustiveness:  int   = 8,
    n_poses:         int   = 10,
    energy_range:    int   = 3,
    use_pubchem:     bool  = False,
    vina_path:       str   = "",
    preferred_ligand:str   = "",
) -> list:
    """
    Batch docking. Prepares receptor once, then docks each ligand.
    Returns list of result dicts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Vina binary ───────────────────────────────────────────────────────────
    if not vina_path:
        vina_path, vmsg = get_vina_binary(str(out_dir))
        if not vina_path:
            return [{"success": False, "error": f"Vina unavailable: {vmsg}"}]
    log(f"Vina ready", "ok")

    # ── Receptor (once) ───────────────────────────────────────────────────────
    center_mode, manual_xyz, prody_sel = _parse_center(center)
    box_size = _parse_box(box)
    section("Receptor Preparation")
    rec_result = prepare_receptor(
        raw_pdb=receptor, wdir=out_dir / "_receptor",
        center_mode=center_mode, manual_xyz=manual_xyz,
        prody_sel=prody_sel, box_size=box_size,
        preferred_ligand=preferred_ligand,
    )
    if not rec_result["success"]:
        return [{"success": False, "error": f"Receptor: {rec_result['error']}"}]
    for msg in rec_result["log"]: log(msg, "ok" if msg.startswith("✓") else "info")
    log(f"Center ({rec_result['cx']:.2f}, {rec_result['cy']:.2f}, {rec_result['cz']:.2f})", "info")

    # ── Batch docking loop ────────────────────────────────────────────────────
    section(f"Batch Docking — {len(smiles_list)} ligand(s)")
    results = []
    csv_rows = []

    for i, (smi, name) in enumerate(smiles_list, 1):
        progress_bar(i - 1, len(smiles_list), label=f"{name}")
        lig_dir = out_dir / name
        lig_dir.mkdir(parents=True, exist_ok=True)

        # ligand prep
        lig_result = prepare_ligand(
            smiles=smi, name=name, ph=ph, wdir=lig_dir / "ligand",
            mode=mode, use_pubchem=use_pubchem,
        )
        if not lig_result["success"]:
            results.append({"success": False, "name": name, "smiles": smi,
                            "error": lig_result["error"]})
            csv_rows.append({"Name": name, "SMILES": smi, "Top Score": "PREP FAILED",
                             "Charge": "", "Status": lig_result["error"]})
            continue

        # docking
        dock_result = run_vina(
            receptor_pdbqt = rec_result["rec_pdbqt"],
            ligand_pdbqt   = lig_result["pdbqt"],
            config_txt     = rec_result["config_txt"],
            vina_path      = vina_path,
            exhaustiveness = exhaustiveness,
            n_modes        = n_poses,
            energy_range   = energy_range,
            wdir           = lig_dir / "docking",
            out_name       = name,
        )
        if not dock_result["success"]:
            results.append({"success": False, "name": name, "smiles": smi,
                            "error": dock_result["error"]})
            csv_rows.append({"Name": name, "SMILES": smi, "Top Score": "DOCK FAILED",
                             "Charge": f"{lig_result['net_charge']:+d}", "Status": "FAILED"})
            continue

        # bond order fix
        pv_sdf = str(lig_dir / f"{name}_poses.sdf")
        fix_sdf_bond_orders(dock_result["out_sdf"], lig_result["prot_smiles"], pv_sdf)
        if not Path(pv_sdf).exists() or Path(pv_sdf).stat().st_size < 10:
            pv_sdf = dock_result["out_sdf"]

        top = dock_result["top_score"]
        results.append({
            "success":     True,
            "name":        name,
            "smiles":      smi,
            "prot_smiles": lig_result["prot_smiles"],
            "charge":      lig_result["net_charge"],
            "top_score":   top,
            "scores":      dock_result["scores"],
            "out_sdf":     pv_sdf,
            "out_pdbqt":   dock_result["out_pdbqt"],
        })
        csv_rows.append({
            "Name": name, "SMILES": smi,
            "Protonated SMILES": lig_result["prot_smiles"],
            "Charge": f"{lig_result['net_charge']:+d}",
            "Top Score (kcal/mol)": f"{top:.2f}" if top else "",
            "Status": "OK",
        })

    progress_bar(len(smiles_list), len(smiles_list), label="done")

    # ── Save CSV summary ──────────────────────────────────────────────────────
    csv_path = str(out_dir / "batch_scores.csv")
    if csv_rows:
        fields = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(csv_rows)
        log(f"Results saved → {csv_path}", "ok")

    # ── Print summary table ───────────────────────────────────────────────────
    ok_results = [r for r in results if r["success"]]
    if ok_results:
        ok_results.sort(key=lambda r: r["top_score"] or 99)
        print(f"\n  {'Name':<20} {'Score':>10}  {'Charge':>6}")
        print(f"  {'-'*20} {'-'*10}  {'-'*6}")
        for r in ok_results:
            print(f"  {r['name']:<20} {r['top_score']:>10.2f}  {r['charge']:>+6d}")
        print()
    log(f"{len(ok_results)}/{len(smiles_list)} ligands docked successfully", "ok")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  FILE I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_smi_file(path: str) -> list:
    """Read .smi file → [(smiles, name), …]. Skips blank lines and comments."""
    pairs = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        smi  = parts[0]
        name = parts[1].replace(" ", "_") if len(parts) > 1 else f"lig_{i}"
        pairs.append((smi, name))
    return pairs


def download_pdb(pdb_id: str, out_path: str, fmt: str = "pdb") -> bool:
    """Download a PDB structure from RCSB."""
    import requests
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{fmt}"
    log(f"Downloading {pdb_id.upper()} from RCSB…", "info")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        Path(out_path).write_bytes(r.content)
        return True
    except Exception as e:
        log(f"Download failed: {e}", "error")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — argparse
# ══════════════════════════════════════════════════════════════════════════════

def _add_common_args(parser):
    parser.add_argument("--receptor", "-r", required=True,
                        help="Receptor PDB/CIF file OR 4-letter PDB ID (auto-downloaded)")
    parser.add_argument("--center",   default="auto",
                        help='"auto" | "x y z" | ProDy selection (default: auto)')
    parser.add_argument("--box",      default="16 16 16",
                        help='Box size in Å: "sx sy sz" or single value (default: 16 16 16)')
    parser.add_argument("--ph",       type=float, default=7.4,
                        help="Target pH for protonation (default: 7.4)")
    parser.add_argument("--mode",     choices=["dimorphite","neutral","pkanet"],
                        default="dimorphite",
                        help="Protonation mode (default: dimorphite)")
    parser.add_argument("--pubchem",  action="store_true",
                        help="Query PubChem for pKa data (pKaNET mode)")
    parser.add_argument("--exhaustiveness", "-e", type=int, default=16,
                        help="Vina exhaustiveness (default: 16)")
    parser.add_argument("--poses",    type=int, default=10,
                        help="Max poses per ligand (default: 10)")
    parser.add_argument("--energy-range", type=int, default=3,
                        help="Vina energy range kcal/mol (default: 3)")
    parser.add_argument("--out",      "-o", default="./docking_results",
                        help="Output directory (default: ./docking_results)")
    parser.add_argument("--vina",     default="",
                        help="Path to Vina binary (auto-downloaded if omitted)")
    parser.add_argument("--preferred-ligand", default="",
                        help="Residue name of preferred co-crystal ligand for grid centering")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ACD_standalone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ╔══════════════════════════════════════════╗
            ║   Anyone Can Dock — Standalone CLI       ║
            ║   AutoDock Vina 1.2.7 + RDKit + Meeko   ║
            ╚══════════════════════════════════════════╝

            Platforms: macOS (Intel/Apple Silicon) · Windows · Linux · Colab
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python ACD_standalone.py setup
              python ACD_standalone.py dock -r 1M17.pdb --smiles "CCO" --name ethanol
              python ACD_standalone.py dock -r 1M17 --smiles "CCO" --name ethanol
              python ACD_standalone.py batch -r 1M17.pdb --smi ligands.smi
              python ACD_standalone.py batch -r 1M17.pdb --smi ligands.smi --mode pkanet
        """),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── setup ────────────────────────────────────────────────────────────────
    sub.add_parser("setup", help="Install missing dependencies")

    # ── dock (single) ─────────────────────────────────────────────────────────
    p_dock = sub.add_parser("dock", help="Dock a single ligand")
    _add_common_args(p_dock)
    p_dock.add_argument("--smiles", "-s", required=True,
                        help='Ligand SMILES string (quote it: "CCO")')
    p_dock.add_argument("--name",   "-n", default="ligand",
                        help="Ligand name / output prefix (default: ligand)")

    # ── batch ──────────────────────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", help="Dock multiple ligands from .smi file")
    _add_common_args(p_batch)
    p_batch.add_argument("--smi",   required=True,
                         help='.smi file — one "SMILES Name" per line')

    # ── info ───────────────────────────────────────────────────────────────────
    sub.add_parser("info", help="Show environment info and dependency status")

    return parser


def _resolve_receptor(receptor_arg: str, out_dir: Path) -> str:
    """If receptor_arg looks like a PDB ID (4 chars), download it."""
    p = Path(receptor_arg)
    if p.exists():
        return str(p)
    # Looks like a PDB ID?
    if len(receptor_arg) == 4 and receptor_arg.isalnum():
        pdb_path = str(out_dir / f"{receptor_arg.upper()}.pdb")
        if not Path(pdb_path).exists():
            ok = download_pdb(receptor_arg, pdb_path, "pdb")
            if not ok:
                ok = download_pdb(receptor_arg, pdb_path, "cif")
        if Path(pdb_path).exists():
            return pdb_path
    raise FileNotFoundError(
        f"Receptor file not found and '{receptor_arg}' is not a valid PDB ID or path."
    )


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "setup":
        setup_command()
        return

    if args.command == "info":
        section("Environment Info")
        log(f"Python {ENV['python']}", "info")
        log(f"OS: {ENV['os_name']} / arch: {ENV['arch']}", "info")
        log(f"Colab: {ENV['is_colab']}  Jupyter: {ENV['is_jupyter']}", "info")
        print()
        section("Dependency Status")
        check_dependencies(verbose=True)
        return

    if args.command is None:
        parser.print_help()
        return

    # ── Resolve receptor ──────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        receptor = _resolve_receptor(args.receptor, out_dir)
    except FileNotFoundError as e:
        log(str(e), "error"); sys.exit(1)

    # ── dock ──────────────────────────────────────────────────────────────────
    if args.command == "dock":
        section(f"Docking: {args.name}")
        result = dock_single(
            receptor         = receptor,
            smiles           = args.smiles,
            name             = args.name,
            out_dir          = str(out_dir),
            ph               = args.ph,
            mode             = args.mode,
            center           = args.center,
            box              = args.box,
            exhaustiveness   = args.exhaustiveness,
            n_poses          = args.poses,
            energy_range     = args.energy_range,
            use_pubchem      = args.pubchem,
            vina_path        = args.vina,
            preferred_ligand = args.preferred_ligand,
        )
        if not result["success"]:
            log(f"Docking failed: {result['error']}", "error"); sys.exit(1)
        section("Output Files")
        log(f"Poses (SDF)  : {result['out_sdf']}", "ok")
        log(f"Poses (PDBQT): {result['out_pdbqt']}", "ok")
        log(f"Receptor PDB : {result['rec_fh']}", "ok")
        log(f"Ligand SMILES: {result['prot_smiles']} (charge {result['charge']:+d})", "info")

    # ── batch ─────────────────────────────────────────────────────────────────
    elif args.command == "batch":
        if not Path(args.smi).exists():
            log(f"SMI file not found: {args.smi}", "error"); sys.exit(1)
        smiles_list = parse_smi_file(args.smi)
        if not smiles_list:
            log("No valid SMILES found in file.", "error"); sys.exit(1)
        log(f"Loaded {len(smiles_list)} ligand(s) from {args.smi}", "info")
        dock_batch(
            receptor         = receptor,
            smiles_list      = smiles_list,
            out_dir          = str(out_dir),
            ph               = args.ph,
            mode             = args.mode,
            center           = args.center,
            box              = args.box,
            exhaustiveness   = args.exhaustiveness,
            n_poses          = args.poses,
            energy_range     = args.energy_range,
            use_pubchem      = args.pubchem,
            vina_path        = args.vina,
            preferred_ligand = args.preferred_ligand,
        )
        section("Output")
        log(f"Results → {out_dir / 'batch_scores.csv'}", "ok")
        log(f"Pose files → {out_dir}/<ligand_name>/", "ok")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
