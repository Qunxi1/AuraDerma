$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Host 'Docker is required to auto-start Qdrant.'
  exit 1
}

$composeFile = Join-Path $root 'docker-compose.yml'
$exists = docker ps -a --format '{{.Names}}' | Select-String -Pattern '^auraderma-qdrant$'
if (-not $exists) {
  docker compose -f $composeFile up -d qdrant
} else {
  $running = docker ps --format '{{.Names}}' | Select-String -Pattern '^auraderma-qdrant$'
  if (-not $running) {
    docker start auraderma-qdrant | Out-Null
  }
}

python -m auraderma chat
