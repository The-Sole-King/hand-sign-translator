# PyInstaller spec: builds a double-click app for the current OS.
#   Windows -> dist/SignReader.exe     macOS -> dist/SignReader.app
#   Linux   -> dist/SignReader
# Run via `python packaging/build.py`, which downloads the hand model first.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

HERE = Path(SPECPATH)
ROOT = HERE.parent
MODELS = [HERE / "assets" / n for n in (
    "hand_landmarker.task", "face_landmarker.task", "face_recognition_sface_2021dec.onnx")]
for model in MODELS:
    if not model.exists():
        raise SystemExit(f"Missing {model}. Run packaging/build.py instead of pyinstaller directly.")

_os = {"win32": "win32", "darwin": "darwin"}.get(sys.platform, "xorg")
PYNPUT_BACKENDS = [f"pynput.keyboard._{_os}", f"pynput.mouse._{_os}", f"pynput._util.{_os}"]

a = Analysis(
    [str(HERE / "launcher.py")],
    pathex=[str(ROOT)],
    # MediaPipe loads its native library from mediapipe/tasks/c via ctypes,
    # which PyInstaller can't see, so collect it (and its data) explicitly.
    binaries=collect_dynamic_libs("mediapipe"),
    datas=collect_data_files("mediapipe") + [(str(m), "assets") for m in MODELS],
    # pynput picks its OS backend at runtime, so PyInstaller can't see it (and
    # can't import it to look on a headless build machine): name it.
    hiddenimports=collect_submodules("mediapipe.tasks.python") + PYNPUT_BACKENDS,
    excludes=["pytest", "IPython"],
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    # A .app bundle: onedir layout, launched from Finder.
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SignReader",
              console=False, argv_emulation=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="SignReader")
    app = BUNDLE(
        coll,
        name="SignReader.app",
        bundle_identifier="com.thesoleking.signreader",
        info_plist={
            "CFBundleDisplayName": "Sign Reader",
            "CFBundleShortVersionString": "1.0.0",
            "NSCameraUsageDescription": "Sign Reader uses the camera to read your hand signs. Video never leaves your computer.",
            "NSHighResolutionCapable": True,
        },
    )
else:
    # A single self-contained executable.
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="SignReader",
              console=False, upx=False)
