"""The GPU smoke test must pass on CPU when asked, and fail cleanly without CUDA."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gs", ROOT / "scripts" / "gpu_smoke_test.py")
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)


def test_cpu_path_passes_and_reports_placement(tmp_path):
    out = tmp_path / "smoke.json"
    res = gs.run(["--device", "cpu", "--steps", "400", "--n-critics", "2", "--out", str(out)])
    assert res["ok"], res["problems"]
    assert res["updates"] > 0 and res["steps_per_second"] > 0
    assert res["placement"]["critic"] == ["cpu"]
    assert out.exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_requesting_cuda_without_cuda_fails_cleanly():
    res = gs.run(["--device", "cuda", "--steps", "10"])
    assert res["ok"] is False
    assert any("CUDA" in p for p in res["problems"])
