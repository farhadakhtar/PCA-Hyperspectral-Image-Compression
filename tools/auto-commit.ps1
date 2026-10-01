# HSI auto-commit: single pass. Safe to run on a schedule.
# Adds all non-ignored changes, commits if dirty, pushes to main.
param(
  [string]$Repo = "D:\Study\Sem 5\AIML lab\PCA-Hyperspectral-Image-Compression"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$porcelain = git status --porcelain
if ([string]::IsNullOrWhiteSpace($porcelain)) { exit 0 }

git add -A
# Re-check: may have been only ignored files
$cached = git diff --cached --name-only
if ([string]::IsNullOrWhiteSpace(($cached | Out-String))) { exit 0 }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$files = ($cached | Select-Object -First 5) -join ", "
$n = ($cached | Measure-Object).Count
if ($n -gt 5) { $files += ", +$($n - 5) more" }
git commit -m "auto: $stamp - $files"
git push origin main
