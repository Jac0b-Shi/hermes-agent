"""Recovery must select evidence from the current session and preserve chronology."""

from types import SimpleNamespace

import pytest

from agent.context_acquisition import format_recovered_context_block, recover_archive_evidence
from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, mutate_worker_execution, register_worker_execution


def test_keyword_recovery_finds_older_current_session_evidence_without_cross_session_leaks(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("current", source="cli")
        db.create_session("other", source="cli")
        db.append_message("current", "user", "mount wait decision")
        for index in range(50):
            db.append_message("current", "assistant", f"unrelated progress {index}")
        db.append_message("other", "user", "mount wait OTHER SESSION SECRET")
        agent = SimpleNamespace(session_id="current", _session_db=db)

        rows = recover_archive_evidence(agent, limit=8, user_text="continue mount wait")

        assert any(row["content"] == "mount wait decision" for row in rows)
        assert not any("OTHER SESSION SECRET" in row["content"] for row in rows)
    finally:
        db.close()


def test_recovered_evidence_keeps_database_chronology():
    block = format_recovered_context_block(
        [{"role": "user", "content": "first request"},
         {"role": "assistant", "content": "later commitment"}],
        max_chars=8000,
    )
    assert block.index("first request") < block.index("later commitment")


def test_worker_recovery_uses_owner_history_and_refuses_foreign_sessions(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("current", "cli")
        db.append_message("current", "user", "preserve the mount wait decision")
        db.archive_and_compact("current", [{"role": "assistant", "content": "compressed state"}])
        db.create_session("other", "cli")
        db.append_message("other", "user", "FOREIGN SECRET")
        epoch = begin_runtime_epoch(db, instance_id="fixture")
        scope = dict(epoch=epoch, execution_id="worker", session_id="current", generation=0)
        register_worker_execution(db, **scope, kind="compute", adoption_secret="secret")
        store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / "outbox")
        try:
            rows = recover_archive_evidence(
                SimpleNamespace(session_id="current", _session_db=store),
                limit=8, user_text="continue mount wait",
            )
            assert any(row["content"] == "preserve the mount wait decision" for row in rows)
            assert not any("FOREIGN SECRET" in row["content"] for row in rows)
            with pytest.raises(RuntimeStoreError, match="permission_denied"):
                store.get_messages_as_conversation("other", include_compacted=True)
        finally:
            if store.failure == "permission_denied":
                # Permission failure is sticky: teardown must not hide that fence.
                with pytest.raises(WorkerPersistenceError, match="permission_denied"):
                    store.close()
            else:
                store.close()
    finally:
        db.close()
