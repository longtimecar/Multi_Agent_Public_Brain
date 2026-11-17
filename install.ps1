# membrain Windows one-click installer
# Run from PowerShell in the repository root:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
param(
    [switch]$NoShortcut,
    [switch]$OpenDashboard
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepoRoot
try {
    Write-Host "==> Setting up membrain in $RepoRoot" -ForegroundColor Cyan

    function Find-Uv {
        $cmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        $candidates = @(
            (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
            (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
            (Join-Path $env:LOCALAPPDATA "uv\uv.exe")
        )
        foreach ($candidate in $candidates) {
            if ($candidate -and (Test-Path $candidate)) { return $candidate }
        }
        return $null
    }

    $uv = Find-Uv
    if (-not $uv) {
        Write-Host "==> uv not found; installing uv..." -ForegroundColor Yellow
        Invoke-RestMethod "https://astral.sh/uv/install.ps1" | Invoke-Expression
        foreach ($dir in @(
            (Join-Path $env:USERPROFILE ".local\bin"),
            (Join-Path $env:USERPROFILE ".cargo\bin")
        )) {
            if ((Test-Path $dir) -and ($env:Path -notlike "*$dir*")) {
                $env:Path = "$dir;$env:Path"
            }
        }
        $uv = Find-Uv
    }
    if (-not $uv) {
        throw "Failed to install or locate uv. Install uv from https://docs.astral.sh/uv/ and re-run install.ps1."
    }
    Write-Host "[ok] Using uv: $uv" -ForegroundColor Green

    Write-Host "==> Installing dependencies (uv sync)..." -ForegroundColor Cyan
    & $uv sync
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Host "[ok] Created .env from .env.example (fill in LLM / Embedding keys later)" -ForegroundColor Green
        }
    } else {
        Write-Host "[ok] Keeping existing .env unchanged" -ForegroundColor Green
    }

    Write-Host "==> Initializing memory store and profile (membrain init)..." -ForegroundColor Cyan
    & $uv run membrain init
    if ($LASTEXITCODE -ne 0) {
        throw "membrain init failed with exit code $LASTEXITCODE"
    }

    if (-not $NoShortcut) {
        Write-Host "==> Creating Windows desktop shortcut (membrain shortcut)..." -ForegroundColor Cyan
        & $uv run membrain shortcut
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Desktop shortcut could not be created automatically. Retry later with: uv run membrain shortcut"
        }
    }

    Write-Host ""
    Write-Host "Setup complete! Logon autostart stays OFF by default." -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Configure API keys in .env or inside Dashboard -> Model settings."
    Write-Host "  2. Launch via the new Desktop shortcut or run: uv run membrain dashboard"
    Write-Host "  3. Connect Claude Code / Codex / OpenClaw in Dashboard -> Agents."
    Write-Host "     (Optional) Enable silent logon autostart in Dashboard -> Settings."

    if ($OpenDashboard) {
        Write-Host "==> Starting dashboard..." -ForegroundColor Cyan
        & $uv run membrain dashboard
    }
}
finally {
    Pop-Location
}
