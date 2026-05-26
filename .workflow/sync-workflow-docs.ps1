param(
    [string]$SourceDir = ".github/workflows",
    [string]$TargetDir = ".workflow/workflows"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $repoRoot $SourceDir
$targetPath = Join-Path $repoRoot $TargetDir

if (-not (Test-Path $sourcePath)) {
    throw "Source workflows folder not found: $sourcePath"
}

if (-not (Test-Path $targetPath)) {
    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
}

# Copy current workflow files from active GitHub Actions directory.
$sourceFiles = Get-ChildItem -Path $sourcePath -File | Where-Object {
    $_.Extension -in @('.yml', '.yaml')
}

$sourceNames = @{}
foreach ($file in $sourceFiles) {
    $destination = Join-Path $targetPath $file.Name
    Copy-Item -Path $file.FullName -Destination $destination -Force
    $sourceNames[$file.Name] = $true
}

# Remove stale files in docs folder that no longer exist in source.
$targetFiles = Get-ChildItem -Path $targetPath -File | Where-Object {
    $_.Extension -in @('.yml', '.yaml')
}

foreach ($file in $targetFiles) {
    if (-not $sourceNames.ContainsKey($file.Name)) {
        Remove-Item -Path $file.FullName -Force
    }
}

Write-Host "Synchronized workflow docs from '$SourceDir' to '$TargetDir'."
