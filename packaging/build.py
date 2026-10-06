"""Build the double-click Sign Reader app for this OS.

    pip install -r requirements.txt pyinstaller
    python packaging/build.py

Output: dist/SignReader.exe (Windows), dist/SignReader.app (macOS) or
dist/SignReader (Linux). Each OS has to build its own app.
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from sign_reader.app import FACE_ID_MODEL_URL, FACE_MODEL_URL, MODEL_URL  # noqa: E402

for name, url in (
    ("hand_landmarker.task", MODEL_URL),
    ("face_landmarker.task", FACE_MODEL_URL),
    ("face_recognition_sface_2021dec.onnx", FACE_ID_MODEL_URL),
):
    model = HERE / "assets" / name
    if not model.exists():
        model.parent.mkdir(exist_ok=True)
        print(f"Downloading {name} to {model} ...")
        urllib.request.urlretrieve(url, model)

for d in ("build", "dist"):
    shutil.rmtree(ROOT / d, ignore_errors=True)
subprocess.run(
    [sys.executable, "-m", "PyInstaller", "--noconfirm", str(HERE / "SignReader.spec")],
    cwd=ROOT,
    check=True,
)
print("\nBuilt:", *sorted(p.name for p in (ROOT / "dist").iterdir()))
