# One-time setup. Run from the repo root (C:\Users\HP\Desktop\Karm).
# If blocked: powershell -ExecutionPolicy Bypass -File setup_env.ps1

Write-Host "Setting up Karm..." -ForegroundColor Cyan

# Preserve your original monolith as a reference, if it's still in the root.
foreach ($f in @("bigram_01.py", "day01_bigram.py")) {
    if (Test-Path $f) {
        Move-Item $f "scripts\day01_bigram_standalone.py" -Force
        Write-Host "Moved $f -> scripts\day01_bigram_standalone.py"
    }
}

# Virtual environment
if (-not (Test-Path ".venv")) { py -m venv .venv }
.\.venv\Scripts\Activate.ps1

# Install (editable, so `import karm` works) + dev tools
pip install --upgrade pip
pip install -e .
pip install pytest

Write-Host "`nDone. Try:" -ForegroundColor Green
Write-Host "  python scripts\day01_bigram.py"
Write-Host "  pytest"
