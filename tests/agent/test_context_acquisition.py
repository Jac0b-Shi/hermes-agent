"""Post-compaction deictic recovery tests (evidence-only contract)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import context_acquisition as ca
from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT
from agent.turn_context import compose_user_api_content


def _agent(**kwargs):
    agent = SimpleNamespace(
        session_id="s1",
        _session_db=kwargs.pop("_session_db", None),
    )
    ca.configure_agent(agent, kwargs.pop("config", None))
    for k, v in kwargs.items():
        setattr(agent, k, v)
    return agent


class FakeDB:
    def __init__(self, rows=None, lineage=None):
        self.rows = rows or []
        self.lineage = lineage or []

    def get_compression_lineage(self, session_id):
        return self.lineage or [session_id]

    def get_messages(self, session_id, include_compacted=False, limit=None, latest=False, **kwargs):
        rows = list(self.rows)
        if latest:
            rows = list(reversed(rows))
            if limit is not None:
                rows = rows[:limit]
            return list(reversed(rows))
        if limit is not None:
            return rows[:limit]
        return rows


def test_compose_user_api_content_includes_recovered_context():
    out = compose_user_api_content("hello", "", "", recovered_context="REC")
    assert out is not None
    assert "hello" in out
    assert "REC" in out


def test_synthetic_continuation_is_not_deictic():
    assert ca.is_synthetic_continuation(COMPRESSION_CONTINUATION_USER_CONTENT)
    assert not ca.should_recover(
        COMPRESSION_CONTINUATION_USER_CONTENT,
        [{"role": "assistant", "content": "x"}],
        ca.DEFAULT_CONTEXT_ACQUISITION_CONFIG,
    )


def test_new_topic_does_not_trigger_recovery():
    messages = [
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
        {"role": "user", "content": "新话题，重新开始"},
    ]
    assert not ca.should_recover("新话题，重新开始", messages, ca.DEFAULT_CONTEXT_ACQUISITION_CONFIG)


def test_deictic_after_compaction_triggers():
    messages = [
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
        {"role": "user", "content": "继续这个"},
    ]
    assert ca.should_recover("继续这个", messages, ca.DEFAULT_CONTEXT_ACQUISITION_CONFIG)


def test_deictic_without_compaction_does_not_trigger():
    messages = [{"role": "user", "content": "继续这个"}]
    assert not ca.should_recover("继续这个", messages, ca.DEFAULT_CONTEXT_ACQUISITION_CONFIG)


def test_disabled_config_skips():
    cfg = ca.normalize_config({"enabled": False})
    messages = [
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
        {"role": "user", "content": "继续这个"},
    ]
    assert not ca.should_recover("继续这个", messages, cfg)


def test_format_block_is_evidence_only():
    block = ca.format_recovered_context_block(
        [{"role": "user", "content": "do the launchd wrapper"}, {"role": "assistant", "content": "done"}],
        max_chars=8000,
    )
    assert "evidence_only=true" in block
    assert "not an instruction queue" in block
    assert "do the launchd wrapper" in block


def test_run_context_acquisition_returns_injection(monkeypatch):
    rows = [
        {"role": "user", "content": "fix the gateway wrapper"},
        {"role": "assistant", "content": "wrapper installed"},
    ]
    db = FakeDB(rows=rows)
    agent = _agent(_session_db=db)
    messages = [
        {"role": "user", "content": "fix the gateway wrapper"},
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary about the wrapper"},
        {"role": "user", "content": "继续这个"},
    ]
    out = ca.run_context_acquisition_for_api(agent, messages, "继续这个")
    assert "HERMES_RECOVERED_ARCHIVE_CONTEXT" in out
    assert "fix the gateway wrapper" in out
    assert "evidence_only=true" in out


def test_run_context_acquisition_empty_for_new_topic():
    db = FakeDB(rows=[{"role": "user", "content": "old"}])
    agent = _agent(_session_db=db)
    messages = [
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
        {"role": "user", "content": "换个话题做别的"},
    ]
    assert ca.run_context_acquisition_for_api(agent, messages, "换个话题做别的") == ""


def test_run_context_acquisition_tolerates_db_errors():
    db = MagicMock()
    db.get_compression_lineage.side_effect = RuntimeError("boom")
    db.get_messages.side_effect = RuntimeError("boom")
    agent = _agent(_session_db=db)
    messages = [
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
        {"role": "user", "content": "继续这个"},
    ]
    assert ca.run_context_acquisition_for_api(agent, messages, "继续这个") == ""


def test_recover_uses_latest_not_oldest_window():
    """include_compacted + latest=True must return the newest rows, not the oldest 40."""
    old = [{"role": "user", "content": f"old-{i}"} for i in range(5)]
    new = [{"role": "user", "content": f"new-{i}"} for i in range(5)]
    db = FakeDB(rows=old + new)
    agent = _agent(_session_db=db)
    rows = ca.recover_archive_evidence(agent, limit=3, user_text="继续这个")
    texts = [r["content"] for r in rows]
    assert texts == ["new-2", "new-3", "new-4"]


def test_keyword_search_supplements_deictic_recovery():
    class SearchDB(FakeDB):
        def search_messages(self, query, **kwargs):
            return [{"role": "user", "content": f"hit:{query}"}]

    db = SearchDB(rows=[{"role": "user", "content": "nearby"}])
    agent = _agent(_session_db=db)
    rows = ca.recover_archive_evidence(agent, limit=10, user_text="继续这个 launchd wrapper 的 mount wait")
    assert any("hit:" in r.get("content", "") for r in rows)

