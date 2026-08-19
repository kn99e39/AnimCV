import importlib.util
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "run_lifter_experiments.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("run_lifter_experiments", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_dataset_fingerprint_records_exact_bytes_and_logical_counts(tmp_path):
    module = _load_module()
    dataset_path = tmp_path / "holdout.json"
    raw = b'{"frames": [1, 2], "sequences": [{"frames": [1]}, {"frames": [2]}]}\n'
    dataset_path.write_bytes(raw)

    fingerprint = module._dataset_fingerprint(
        dataset_path, {"frames": [1, 2], "sequences": [{"frames": [1]}, {"frames": [2]}]},
    )

    assert fingerprint == {
        "path": str(dataset_path),
        "sha256": "d491e325148e4ab1cdfd9d921673dbc44ea1a21cb1f979f1293c6b73bd9f791d",
        "byte_size": len(raw),
        "frame_count": 2,
        "sequence_count": 2,
    }
