"""Checks on notebooks/colab_train.ipynb (P-008): it must not pin the old tag."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "colab_train.ipynb"


def _cells():
    nb = json.loads(NB.read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def test_no_cell_checks_out_old_tag():
    for _, src in _cells():
        assert not re.search(r"git checkout\s+lin-env-v1", src)


def test_clone_cell_uses_code_ref_and_prints_commit():
    cells = _cells()
    defs = [s for t, s in cells if t == "code" and re.search(r"^CODE_REF\s*=\s*\"main\"", s, re.MULTILINE)]
    assert defs, "no code cell defines CODE_REF = \"main\""
    clone = [s for t, s in cells if t == "code" and "git clone" in s]
    assert len(clone) == 1
    assert re.search(r"git checkout\s+\{CODE_REF\}", clone[0])
    assert "git rev-parse HEAD" in clone[0]


def test_table2_outputs_separate_from_pilot():
    """TE-025: Table 2 runs are trained independently of the A-39 pilot, so their --out folders
    must never be the pilot folders (uavmec/pos_*_sic_seed0), or --resume would continue a pilot."""
    outs = []
    for t, src in _cells():
        if t == "code":
            outs += re.findall(r"train\.py[^\n]*--out\s+(\S+)", src)
    assert len(outs) == 6
    for o in outs:
        assert "/MyDrive/uavmec/table2/" in o, o


def test_mentioned_scripts_exist():
    names = set()
    for _, src in _cells():
        names.update(re.findall(r"scripts/(\w+)\.py", src))
    assert names
    for n in names:
        assert (ROOT / "scripts" / f"{n}.py").exists(), n
