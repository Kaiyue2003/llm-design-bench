$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed or is not available on PATH."
}

$projectDirectory = Split-Path -Parent $PSScriptRoot
$cliArguments = @($args)
if ($cliArguments.Count -eq 0) {
    $cliArguments = @("--help")
}
docker compose --project-directory $projectDirectory -f (Join-Path $projectDirectory "compose.yaml") run --rm benchmark @cliArguments
exit $LASTEXITCODE
