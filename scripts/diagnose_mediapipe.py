"""Collect everything needed to identify why MediaPipe's native bindings fail.

    D:\\SAI\\.venv\\Scripts\\python.exe scripts\\diagnose_mediapipe.py

Prints a report. Paste the whole thing. Reads only, changes nothing.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import platform
import sys
import traceback
from pathlib import Path

LINE = "-" * 72


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def main() -> int:
    section("INTERPRETER")
    print("executable :", sys.executable)
    print("version    :", sys.version.replace("\n", " "))
    print("arch       :", platform.machine(), platform.architecture()[0])
    print("platform   :", platform.platform())
    if sys.platform == "win32":
        print("win32_ver  :", platform.win32_ver())
        try:
            print("edition    :", platform.win32_edition())
            print("is_iot     :", platform.win32_is_iot())
        except Exception as exc:
            print("edition    : unavailable", exc)

    section("PACKAGE VERSIONS")
    from importlib.metadata import PackageNotFoundError, distributions, version
    watch = ["mediapipe", "numpy", "protobuf", "flatbuffers", "attrs",
             "opencv-contrib-python", "opencv-python", "opencv-python-headless",
             "opencv-contrib-python-headless", "sentencepiece", "sounddevice",
             "jax", "jaxlib", "absl-py", "matplotlib"]
    for name in watch:
        try:
            print(f"{name:<32} {version(name)}")
        except PackageNotFoundError:
            print(f"{name:<32} not installed")
    installed_opencv = sorted({
        d.metadata["Name"] for d in distributions()
        if (d.metadata["Name"] or "").lower().startswith("opencv")})
    print("\nopencv distributions present:", installed_opencv)

    section("VISUAL C++ RUNTIME")
    if sys.platform == "win32":
        for dll in ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
                    "msvcp140_1.dll", "concrt140.dll", "ucrtbase.dll"]:
            try:
                handle = ctypes.CDLL(dll)
                print(f"  ok      {dll}  -> {getattr(handle, '_name', dll)}")
            except OSError as exc:
                print(f"  MISSING {dll}  -> {exc}")
    else:
        print("  not Windows, skipped")

    section("MEDIA FOUNDATION (OpenCV video backends need these)")
    if sys.platform == "win32":
        for dll in ["mf.dll", "mfplat.dll", "mfreadwrite.dll", "mfcore.dll"]:
            try:
                ctypes.CDLL(dll)
                print(f"  ok      {dll}")
            except OSError as exc:
                print(f"  MISSING {dll}  -> {exc}")
        print("  Note: Windows Server and N editions ship without these unless the")
        print("        Media Foundation / Media Feature Pack feature is installed.")
    else:
        print("  not Windows, skipped")

    section("STEP-BY-STEP IMPORT")
    for mod in ["numpy", "cv2", "google.protobuf", "flatbuffers",
                "mediapipe", "mediapipe.python",
                "mediapipe.python._framework_bindings",
                "mediapipe.tasks", "mediapipe.tasks.python",
                "mediapipe.tasks.python.vision"]:
        try:
            m = importlib.import_module(mod)
            where = getattr(m, "__file__", "(namespace)")
            print(f"  ok      {mod}\n            {where}")
        except Exception as exc:
            print(f"  FAILED  {mod}")
            print(f"            {type(exc).__name__}: {exc}")
            break

    section("NATIVE EXTENSION FILE")
    try:
        import mediapipe
        root = Path(mediapipe.__file__).parent
    except Exception:
        for p in sys.path:
            cand = Path(p) / "mediapipe"
            if cand.is_dir():
                root = cand
                break
        else:
            print("  mediapipe package directory not found on sys.path")
            return 1
    print("  package dir:", root)
    pyds = sorted(root.rglob("_framework_bindings*.pyd")) + \
           sorted(root.rglob("_framework_bindings*.so"))
    if not pyds:
        print("  NO _framework_bindings binary found. The wheel is incomplete.")
    for pyd in pyds:
        print(f"  {pyd}  ({pyd.stat().st_size / 1e6:.1f} MB)")
        if sys.platform == "win32":
            try:
                ctypes.WinDLL(str(pyd))
                print("    ctypes.WinDLL loaded it cleanly "
                      "(so the failure is in Python-level init, not the DLL itself)")
            except OSError as exc:
                print(f"    ctypes.WinDLL failed: winerror={getattr(exc, 'winerror', None)} {exc}")

    section("DLL SEARCH PATH")
    print("  PATH entries containing 'python', 'venv', 'ffmpeg' or 'cuda':")
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        low = entry.lower()
        if any(k in low for k in ("python", "venv", "ffmpeg", "cuda", "nvidia")):
            print("   ", entry)

    section("FULL TRACEBACK OF THE FAILING IMPORT")
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        print("  It imported successfully in this process. "
              "If the pipeline still fails, the venv it runs in differs from this one.")
    except Exception:
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
