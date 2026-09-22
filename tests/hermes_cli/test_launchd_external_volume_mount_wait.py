"""External-volume launchd graft: mount-wait only; osascript stays first executable."""
from __future__ import annotations

from pathlib import Path

from hermes_cli.gateway_launchd import _external_volume_wait_prefix, launchd_program_arguments


def test_no_wait_for_boot_volume_python():
    assert _external_volume_wait_prefix(["/usr/bin/python3", "-m", "x"]) == ""


def test_wait_prefix_when_python_on_external_volume():
    prefix = _external_volume_wait_prefix(["/Volumes/Data/venv/bin/python", "-m", "x"])
    assert "/Volumes/Data" in prefix
    assert "not mounted" in prefix
    assert "sleep 5" in prefix


def test_program_arguments_stay_osascript_on_external_volume(tmp_path: Path):
    args = launchd_program_arguments(
        ["/Volumes/Data/venv/bin/python", "-m", "hermes_cli.main", "gateway", "run"],
        tmp_path / "out.log",
        tmp_path / "err.log",
    )
    assert args[0] == "/usr/bin/osascript"
    assert args[1] == "-e"
    script = args[2]
    assert "do shell script" in script
    # Never a zsh trampoline as the job's first executable (Local Network Privacy, #71206).
    assert "/bin/zsh" not in args[0]
    # Mount-wait is present in the inner shell only.
    assert "not mounted" in script
    assert "exec " in script


def test_program_arguments_no_wait_on_boot_volume(tmp_path: Path):
    args = launchd_program_arguments(
        ["/usr/bin/python3", "-m", "hermes_cli.main", "gateway", "run"],
        tmp_path / "out.log",
        tmp_path / "err.log",
    )
    assert args[0] == "/usr/bin/osascript"
    assert "not mounted" not in args[2]
