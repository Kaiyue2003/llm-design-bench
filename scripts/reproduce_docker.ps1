$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed or is not available on PATH."
}

$projectDirectory = Split-Path -Parent $PSScriptRoot
$cliArguments = @($args)

$containerUser = $env:LLMDM_CONTAINER_USER
if ([string]::IsNullOrEmpty($containerUser)) {
    if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) {
        # Docker Desktop's Linux-container bind mounts use Windows host access.
        $containerUser = "1000:1000"
    } else {
        $callerUid = & id -u
        if ($LASTEXITCODE -ne 0) { throw "Could not determine the caller UID." }
        $callerGid = & id -g
        if ($LASTEXITCODE -ne 0) { throw "Could not determine the caller GID." }
        $containerUser = "${callerUid}:${callerGid}"
    }
}
if ($containerUser -notmatch '^[0-9]+:[0-9]+$') {
    throw "LLMDM_CONTAINER_USER must be a numeric UID:GID pair."
}

function Resolve-BenchmarkDirectory {
    param([string]$ConfiguredPath, [string]$DefaultRelativePath)
    if ([string]::IsNullOrEmpty($ConfiguredPath)) {
        $ConfiguredPath = $DefaultRelativePath
    }
    if (-not [IO.Path]::IsPathRooted($ConfiguredPath)) {
        $ConfiguredPath = Join-Path $projectDirectory $ConfiguredPath
    }
    $directoryPath = [IO.Path]::GetFullPath($ConfiguredPath)
    # This never overwrites files or changes permissions of existing directories.
    return [IO.Directory]::CreateDirectory($directoryPath).FullName
}

$resultsDirectory = Resolve-BenchmarkDirectory $env:RESULTS_DIR "results/docker"
$assetsDirectory = Resolve-BenchmarkDirectory $env:ASSETS_DIR "assets"
$service = "benchmark"
$build = $false
while ($cliArguments.Count -gt 0) {
    if ($cliArguments[0] -eq "--smoke") {
        $service = "smoke"
    } elseif ($cliArguments[0] -eq "--build") {
        $build = $true
    } elseif ($cliArguments[0] -eq "--") {
        $cliArguments = @($cliArguments | Select-Object -Skip 1)
        break
    } else {
        break
    }
    $cliArguments = @($cliArguments | Select-Object -Skip 1)
}
if ($service -eq "benchmark" -and $cliArguments.Count -eq 0) {
    $cliArguments = @("--help")
}
$dockerArguments = @("compose", "--project-directory", $projectDirectory,
    "-f", (Join-Path $projectDirectory "compose.yaml"), "run", "--rm")
if ($build) { $dockerArguments += "--build" }
$dockerArguments += $service
$dockerArguments += $cliArguments

# Do not leave temporary wrapper defaults in an interactive PowerShell session.
$originalEnvironment = @{}
foreach ($name in @("LLMDM_CONTAINER_USER", "RESULTS_DIR", "ASSETS_DIR")) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
    $env:LLMDM_CONTAINER_USER = $containerUser
    $env:RESULTS_DIR = $resultsDirectory
    $env:ASSETS_DIR = $assetsDirectory
    & docker @dockerArguments
    $dockerExitCode = $LASTEXITCODE
} finally {
    foreach ($name in $originalEnvironment.Keys) {
        if ($null -eq $originalEnvironment[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], "Process")
        }
    }
}
exit $dockerExitCode
