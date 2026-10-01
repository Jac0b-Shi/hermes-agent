"""External-volume launchd graft: mount-wait only; osascript stays first executable.

Also covers launchd stdio placement: xpcproxy cannot open StandardOutPath on
/Volumes (EPERM → fake posix_spawn 78), so those keys must stay on the boot
volume whenever the gateway's own logs live on an external volume.
"""
from __future__ import annotations

from pathlib import Path

from hermes_cli.gateway_launchd import (
    _external_volume_wait_prefix,
    _launchd_stdio_paths,
    launchd_program_arguments,
)


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


def test_launchd_stdio_stays_off_external_volume(tmp_path: Path):
    app_out = Path("/Volumes/Data/Library/Application Support/Hermes Agent/logs/gateway.log")
    app_err = app_out.with_name("gateway.error.log")
    launchd_out, launchd_err = _launchd_stdio_paths(app_out, app_err)
    # xpcproxy opens these before posix_spawn — never on /Volumes (EPERM → exit 78).
    assert not str(launchd_out).startswith("/Volumes/")
    assert not str(launchd_err).startswith("/Volumes/")
    assert launchd_out.name == "gateway.log" and launchd_err.name == "gateway.error.log"


def test_launchd_stdio_keeps_app_paths_on_boot_volume(tmp_path: Path):
    app_out = tmp_path / "logs" / "gateway.log"
    app_err = tmp_path / "logs" / "gateway.error.log"
    launchd_out, launchd_err = _launchd_stdio_paths(app_out, app_err)
    # Same paths: shell redirects and launchd stdio may share the boot-volume files.
    assert launchd_out == app_out
    assert launchd_err == app_err
