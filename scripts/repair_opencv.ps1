<#
    Repairs the OpenCV / MediaPipe DLL conflict on Windows.

    Symptom:
        ImportError: DLL load failed while importing _framework_bindings:
        A dynamic link library (DLL) initialization routine failed.

    Cause:
        opencv-python, opencv-python-headless and opencv-contrib-python all
        install into the same site-packages\cv2 directory. MediaPipe is built
        against opencv-contrib-python. When another one is installed on top,
        the native DLLs no longer match and MediaPipe's bindings fail to
        initialise, even though `import mediapipe` may still appear to work.

    Run from the repository root:
        .\scripts\repair_opencv.ps1
#>

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv not found at $py. Run scripts\bootstrap.ps1 first." }

Write-Host "Currently installed OpenCV distributions:" -ForegroundColor Cyan
& $py -m pip list | Select-String -Pattern "opencv"

Write-Host "`nRemoving every OpenCV distribution ..." -ForegroundColor Cyan
& $py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>$null

# pip can leave the cv2 directory behind when two packages shared it.
$cv2 = & $py -c "import sysconfig,os;print(os.path.join(sysconfig.get_paths()['purelib'],'cv2'))"
if (Test-Path $cv2) {
    Write-Host "Removing leftover $cv2 ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $cv2
}

Write-Host "`nReinstalling the single distribution MediaPipe expects ..." -ForegroundColor Cyan
& $py -m pip install --no-cache-dir opencv-contrib-python==4.10.0.84
& $py -m pip install --no-cache-dir --force-reinstall mediapipe==0.10.21

Write-Host "`nVerifying the exact imports S2 performs ..." -ForegroundColor Cyan
& $py -c "import cv2; print('cv2', cv2.__version__); from mediapipe.tasks.python import vision; import mediapipe as mp; print('mediapipe', mp.__version__, 'tasks vision OK')"

Write-Host "`nRunning the environment check ..." -ForegroundColor Cyan
& $py -m archery doctor
