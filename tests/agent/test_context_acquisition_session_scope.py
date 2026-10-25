"""Recovery must select evidence from the current session and preserve chronology."""

from types import SimpleNamespace

from agent.context_acquisition import format_recovered_context_block, recover_archive_evidence
from hermes_state import SessionDB


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
        assert all(row["session_id"] == "current" for row in rows)
    finally:
        db.close()


def test_recovered_evidence_keeps_database_chronology():
    block = format_recovered_context_block(
        [{"role": "user", "content": "first request"},
         {"role": "assistant", "content": "later commitment"}],
        max_chars=8000,
    )
    assert block.index("first request") < block.index("later commitment")
