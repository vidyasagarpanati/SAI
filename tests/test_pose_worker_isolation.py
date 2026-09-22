"""The pose worker exists to keep MediaPipe away from pyarrow. Keep it that way."""
import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "src" / "archery" / "pose_worker.py"
ALLOWED_TOP = {"argparse", "json", "sys", "time", "pathlib", "__future__",
               "numpy", "cv2", "mediapipe"}
FORBIDDEN = {"pandas", "pyarrow", "langgraph", "archery"}


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


def test_worker_imports_only_safe_modules():
    found = set(_imports(ast.parse(WORKER.read_text())))
    assert not (found & FORBIDDEN), f"pose_worker imports {found & FORBIDDEN}"
    assert found <= ALLOWED_TOP, f"unexpected imports: {found - ALLOWED_TOP}"


def test_s02_does_not_import_mediapipe_in_process():
    s02 = WORKER.parent / "steps" / "s02_pose.py"
    found = set(_imports(ast.parse(s02.read_text())))
    assert "mediapipe" not in found, "S2 must delegate MediaPipe to the worker process"
