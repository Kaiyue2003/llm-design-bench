"""Exercise host wrappers with stub Docker; no package, daemon, or image imports.

Run this file alone with ``pytest --noconftest tests/test_docker_wrappers.py``.
POSIX shell tests run on POSIX hosts; PowerShell tests run when pwsh or Windows
PowerShell is installed. These verify argument/environment handling, not Docker
mount permissions or the container image itself.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
WRAPPER_ENVIRONMENT = ("RESULTS_DIR", "ASSETS_DIR", "LLMDM_CONTAINER_USER")

_POWERSHELL_DRIVER = r"""
$ErrorActionPreference = "Stop"
function global:id {
    [System.IO.File]::AppendAllText($env:WRAPPER_TEST_ID_LOG, "called`n")
    if ($env:WRAPPER_TEST_FORBID_ID -eq "1") {
        throw "The wrapper must not invoke id in this case"
    }
    if ($args.Count -ne 1) { throw "Unexpected id arguments" }
    if ($args[0] -eq "-u") { $env:WRAPPER_TEST_UID }
    elseif ($args[0] -eq "-g") { $env:WRAPPER_TEST_GID }
    else { throw "Unexpected id arguments" }
    $global:LASTEXITCODE = 0
}
if ($env:WRAPPER_TEST_NO_DOCKER -ne "1") {
    function global:docker {
        $record = [ordered]@{
            argv = @($args)
            environment = [ordered]@{
                RESULTS_DIR = $env:RESULTS_DIR
                ASSETS_DIR = $env:ASSETS_DIR
                LLMDM_CONTAINER_USER = $env:LLMDM_CONTAINER_USER
            }
        }
        $json = ConvertTo-Json -InputObject $record -Depth 4 -Compress
        [System.IO.File]::AppendAllText($env:WRAPPER_TEST_CAPTURE, $json + "`n")
        $global:LASTEXITCODE = [int]$env:WRAPPER_TEST_DOCKER_EXIT
    }
}
$wrapperArguments = @(ConvertFrom-Json -InputObject $env:WRAPPER_TEST_ARGUMENTS)
& $env:WRAPPER_TEST_SCRIPT @wrapperArguments
$wrapperExitCode = $LASTEXITCODE
$afterEnvironment = [ordered]@{
    RESULTS_DIR = $env:RESULTS_DIR
    ASSETS_DIR = $env:ASSETS_DIR
    LLMDM_CONTAINER_USER = $env:LLMDM_CONTAINER_USER
}
$afterJson = ConvertTo-Json -InputObject $afterEnvironment -Compress
[System.IO.File]::WriteAllText($env:WRAPPER_TEST_AFTER_CAPTURE, $afterJson)
exit $wrapperExitCode
"""

_DOCKER_STUB = r"""
import json
import os
import sys

record = {
    "argv": sys.argv[1:],
    "environment": {
        key: os.environ.get(key)
        for key in ("RESULTS_DIR", "ASSETS_DIR", "LLMDM_CONTAINER_USER")
    },
}
with open(os.environ["WRAPPER_TEST_CAPTURE"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
sys.exit(int(os.environ["WRAPPER_TEST_DOCKER_EXIT"]))
"""

_ID_STUB = r"""
import os
import sys

with open(os.environ["WRAPPER_TEST_ID_LOG"], "a", encoding="utf-8") as stream:
    stream.write("called\n")
if os.environ["WRAPPER_TEST_FORBID_ID"] == "1":
    sys.exit(91)
names = {"-u": "WRAPPER_TEST_UID", "-g": "WRAPPER_TEST_GID"}
if len(sys.argv) != 2 or sys.argv[1] not in names:
    sys.exit(92)
print(os.environ[names[sys.argv[1]]])
"""


def _posix_stub(directory: Path, name: str, source: str) -> None:
    module = directory / f"{name}_stub.py"
    module.write_text(source, encoding="utf-8")
    executable = directory / name
    executable.write_text(
        "#!/bin/sh\n"
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(module))} "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)


@dataclass
class WrapperHarness:
    kind: str
    shell: str
    project: Path
    elsewhere: Path
    stub_directory: Path
    capture: Path
    after_capture: Path
    id_log: Path
    driver: Path

    @property
    def default_user(self) -> str:
        if os.name == "nt":
            return "1000:1000"
        return f"{os.getuid()}:{os.getgid()}"

    def run(
        self,
        arguments: tuple[str, ...] = (),
        *,
        environment: dict[str, str] | None = None,
        docker_exit: int = 0,
        forbid_id: bool = False,
        no_docker: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        for name in WRAPPER_ENVIRONMENT:
            env.pop(name, None)
        env.update(environment or {})
        env.update(
            WRAPPER_TEST_CAPTURE=str(self.capture),
            WRAPPER_TEST_AFTER_CAPTURE=str(self.after_capture),
            WRAPPER_TEST_ID_LOG=str(self.id_log),
            WRAPPER_TEST_DOCKER_EXIT=str(docker_exit),
            WRAPPER_TEST_FORBID_ID="1" if forbid_id or os.name == "nt" else "0",
            WRAPPER_TEST_NO_DOCKER="1" if no_docker else "0",
            WRAPPER_TEST_UID=self.default_user.split(":")[0],
            WRAPPER_TEST_GID=self.default_user.split(":")[1],
        )
        if self.kind == "powershell":
            env["WRAPPER_TEST_SCRIPT"] = str(
                self.project / "scripts/reproduce_docker.ps1"
            )
            env["WRAPPER_TEST_ARGUMENTS"] = json.dumps(arguments)
            if no_docker:
                # No external Docker can accidentally be selected by Get-Command.
                env["PATH"] = str(self.stub_directory)
            command = [
                self.shell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.driver),
            ]
        else:
            if no_docker:
                (self.stub_directory / "docker").unlink()
                # Keep only host utilities needed by ordinary POSIX wrappers.
                for name in ("dirname", "mkdir", "realpath", "readlink"):
                    utility = shutil.which(name)
                    if utility:
                        (self.stub_directory / name).symlink_to(utility)
                env["PATH"] = str(self.stub_directory)
            else:
                env["PATH"] = (
                    str(self.stub_directory) + os.pathsep + env.get("PATH", "")
                )
            command = [
                self.shell,
                str(self.project / "scripts/reproduce_docker.sh"),
                *arguments,
            ]
        return subprocess.run(
            command,
            cwd=self.elsewhere,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def record(self) -> dict:
        records = [json.loads(line) for line in self.capture.read_text().splitlines()]
        assert len(records) == 1, "The wrapper must dispatch exactly one Docker command"
        return records[0]


@pytest.fixture(params=("posix", "powershell"))
def wrapper(request, tmp_path) -> WrapperHarness:
    if request.param == "posix":
        if os.name == "nt":
            pytest.skip("POSIX wrapper requires a POSIX host")
        shell = shutil.which("sh")
    else:
        shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip(f"{request.param} shell is unavailable")
    project = tmp_path / "checkout with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in ("reproduce_docker.sh", "reproduce_docker.ps1"):
        shutil.copyfile(REPOSITORY / "scripts" / name, scripts / name)
    elsewhere = tmp_path / "unrelated working directory"
    elsewhere.mkdir()
    stub_directory = tmp_path / "stub commands"
    stub_directory.mkdir()
    if request.param == "posix":
        _posix_stub(stub_directory, "docker", _DOCKER_STUB)
        _posix_stub(stub_directory, "id", _ID_STUB)
    driver = tmp_path / "invoke wrapper.ps1"
    driver.write_text(_POWERSHELL_DRIVER, encoding="utf-8")
    return WrapperHarness(
        request.param,
        shell,
        project,
        elsewhere,
        stub_directory,
        tmp_path / "docker-calls.jsonl",
        tmp_path / "after-wrapper-environment.json",
        tmp_path / "id-calls.txt",
        driver,
    )


def _assert_invocation(wrapper, record, *, service, build=False, arguments=()):
    expected = [
        "compose",
        "--project-directory",
        str(wrapper.project),
        "-f",
        str(wrapper.project / "compose.yaml"),
        "run",
        "--rm",
    ]
    if build:
        expected.append("--build")
    assert record["argv"] == [*expected, service, *arguments]


def test_default_help_resolves_project_and_creates_caller_owned_directories(wrapper):
    completed = wrapper.run()
    assert completed.returncode == 0, completed.stderr
    record = wrapper.record()
    _assert_invocation(wrapper, record, service="benchmark", arguments=("--help",))
    expected = {
        "RESULTS_DIR": wrapper.project / "results/docker",
        "ASSETS_DIR": wrapper.project / "assets",
    }
    for name, path in expected.items():
        assert path.is_dir()
        assert record["environment"][name] == str(path)
        assert not (wrapper.elsewhere / path.name).exists()
        if os.name != "nt":
            assert path.stat().st_uid == os.getuid()
    assert record["environment"]["LLMDM_CONTAINER_USER"] == wrapper.default_user
    if os.name == "nt":
        assert not wrapper.id_log.exists(), "Windows must never require id.exe"


@pytest.mark.parametrize(
    "arguments,service,build,forwarded",
    [
        (("--smoke",), "smoke", False, ()),
        (("--build",), "benchmark", True, ("--help",)),
        (("--build", "--smoke"), "smoke", True, ()),
        (("--smoke", "--build"), "smoke", True, ()),
        (
            ("--smoke", "--results-root", "/results/custom path"),
            "smoke",
            False,
            ("--results-root", "/results/custom path"),
        ),
        (
            ("run", "--build", "--smoke"),
            "benchmark",
            False,
            ("run", "--build", "--smoke"),
        ),
    ],
)
def test_leading_switches_select_service_without_replacing_smoke_defaults(
    wrapper, arguments, service, build, forwarded
):
    completed = wrapper.run(arguments)
    assert completed.returncode == 0, completed.stderr
    _assert_invocation(
        wrapper, wrapper.record(), service=service, build=build, arguments=forwarded
    )


def test_custom_paths_and_arguments_remain_literal(wrapper, tmp_path):
    assets = tmp_path / "external assets with spaces"
    results = wrapper.project / "custom results with spaces"
    results.mkdir()
    sentinel = results / "existing-result.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    arguments = (
        "run",
        "--plan",
        "data plan.json",
        "a; touch INJECTED",
        "$(touch INJECTED)",
        "literal'\"&|$HOME",
    )
    completed = wrapper.run(
        arguments,
        environment={"RESULTS_DIR": results.name, "ASSETS_DIR": str(assets)},
    )
    assert completed.returncode == 0, completed.stderr
    record = wrapper.record()
    _assert_invocation(wrapper, record, service="benchmark", arguments=arguments)
    assert record["environment"]["RESULTS_DIR"] == str(results)
    assert record["environment"]["ASSETS_DIR"] == str(assets)
    assert assets.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (wrapper.elsewhere / "INJECTED").exists()


@pytest.mark.parametrize("user", ("0:0", "1234:5678"))
def test_explicit_user_override_is_preserved_without_calling_id(wrapper, user):
    completed = wrapper.run(environment={"LLMDM_CONTAINER_USER": user}, forbid_id=True)
    assert completed.returncode == 0, completed.stderr
    assert wrapper.record()["environment"]["LLMDM_CONTAINER_USER"] == user
    assert not wrapper.id_log.exists()
    if os.name != "nt":
        for directory in (
            wrapper.project / "results/docker",
            wrapper.project / "assets",
        ):
            assert directory.stat().st_uid == os.getuid()


@pytest.mark.parametrize(
    "user", ("alice", "1000", "1:", "-1:2", "1:2:3", " 1:2", "1:2 ")
)
def test_invalid_user_fails_before_directory_creation_or_docker(wrapper, user):
    completed = wrapper.run(environment={"LLMDM_CONTAINER_USER": user})
    assert completed.returncode != 0
    assert not wrapper.capture.exists()
    assert not (wrapper.project / "results").exists()
    assert not (wrapper.project / "assets").exists()


def test_empty_user_environment_uses_the_same_default_as_unset(wrapper):
    completed = wrapper.run(environment={"LLMDM_CONTAINER_USER": ""})
    assert completed.returncode == 0, completed.stderr
    assert (
        wrapper.record()["environment"]["LLMDM_CONTAINER_USER"] == wrapper.default_user
    )
    if os.name == "nt":
        assert not wrapper.id_log.exists()


def test_posix_identity_lookup_failure_stops_before_writes_or_docker(wrapper):
    if os.name == "nt":
        pytest.skip("Windows PowerShell does not invoke POSIX id")
    completed = wrapper.run(forbid_id=True)
    assert completed.returncode != 0
    assert wrapper.id_log.exists()
    assert not wrapper.capture.exists()
    assert not (wrapper.project / "results").exists()
    assert not (wrapper.project / "assets").exists()


@pytest.mark.parametrize("docker_exit", (0, 23))
@pytest.mark.parametrize("set_environment", (False, True))
def test_powershell_restores_original_environment_even_when_docker_fails(
    wrapper, docker_exit, set_environment
):
    if wrapper.kind != "powershell":
        pytest.skip("Only PowerShell wrappers can affect their caller's environment")
    configured = (
        {
            "LLMDM_CONTAINER_USER": "0:0",
            "RESULTS_DIR": "relative caller results",
            "ASSETS_DIR": "relative caller assets",
        }
        if set_environment
        else {}
    )
    completed = wrapper.run(environment=configured, docker_exit=docker_exit)
    assert completed.returncode == docker_exit, completed.stderr
    restored = json.loads(wrapper.after_capture.read_text())
    assert restored == {name: configured.get(name) for name in WRAPPER_ENVIRONMENT}


def test_environment_is_not_replaced_by_dotenv_file(wrapper):
    (wrapper.project / ".env").write_text(
        "LLMDM_CONTAINER_USER=not-a-user\nRESULTS_DIR=wrong-results\nASSETS_DIR=wrong-assets\n",
        encoding="utf-8",
    )
    completed = wrapper.run()
    assert completed.returncode == 0, completed.stderr
    environment = wrapper.record()["environment"]
    assert environment["LLMDM_CONTAINER_USER"] == wrapper.default_user
    assert environment["RESULTS_DIR"] == str(wrapper.project / "results/docker")
    assert environment["ASSETS_DIR"] == str(wrapper.project / "assets")


def test_existing_file_blocks_nested_directory_creation_without_docker(wrapper):
    blocked = wrapper.project / "blocked"
    blocked.write_text("do not replace", encoding="utf-8")
    completed = wrapper.run(environment={"RESULTS_DIR": "blocked/nested results"})
    assert completed.returncode != 0
    assert blocked.read_text(encoding="utf-8") == "do not replace"
    assert not wrapper.capture.exists()


def test_missing_docker_is_reported_without_creating_directories(wrapper):
    completed = wrapper.run(no_docker=True)
    assert completed.returncode != 0
    assert "docker" in (completed.stdout + completed.stderr).lower()
    assert not wrapper.capture.exists()
    assert not (wrapper.project / "results").exists()
    assert not (wrapper.project / "assets").exists()


def test_docker_exit_code_reaches_the_caller(wrapper):
    completed = wrapper.run(("methods",), docker_exit=23)
    assert completed.returncode == 23, completed.stderr
    _assert_invocation(
        wrapper, wrapper.record(), service="benchmark", arguments=("methods",)
    )
