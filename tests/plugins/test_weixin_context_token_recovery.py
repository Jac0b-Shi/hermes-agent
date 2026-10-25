"""A failed send owns only its token generation, never a newer inbound token."""

from unittest.mock import AsyncMock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.weixin import adapter as wx


def connected_adapter(tmp_path):
    adapter = wx.WeixinAdapter(PlatformConfig(enabled=True, token="test-token", extra={"account_id": "acct"}))
    adapter._send_session = object()
    adapter._token_store = wx.ContextTokenStore(str(tmp_path))
    adapter._send_chunk_delay_seconds = 0
    adapter.MAX_MESSAGE_LENGTH = 50
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("stale_response", [False, True])
@pytest.mark.parametrize(("initial_token", "renewed_token"), [(None, "new"), ("old", "old"), ("old", "new")])
async def test_new_inbound_token_survives_old_request_and_reaches_later_chunks(
    tmp_path, monkeypatch, media, stale_response, initial_token, renewed_token,
):
    adapter = connected_adapter(tmp_path)
    if initial_token:
        await adapter._token_store.set("acct", "peer", initial_token)
    tokens = []

    async def reply(request):
        payload = await request.json()
        tokens.append(payload["msg"].get("context_token"))
        if len(tokens) == 1:
            await adapter._token_store.set("acct", "peer", renewed_token)
            if stale_response:
                return web.json_response({"ret": -2, "errmsg": "prepare failed"})
        return web.json_response({"ret": 0})

    if media:
        monkeypatch.setattr(wx, "_get_upload_url", AsyncMock(return_value={"upload_full_url": "https://cdn.example/upload"}))
        monkeypatch.setattr(wx, "_upload_ciphertext", AsyncMock(return_value="encrypted"))
        document = tmp_path / "report.pdf"
        document.write_bytes(b"%PDF-1.4")
    app = web.Application()
    app.router.add_post("/" + wx.EP_SEND_MESSAGE, reply)
    async with TestServer(app) as server, aiohttp.ClientSession(trust_env=False) as session:
        adapter._base_url = str(server.make_url("/"))
        adapter._send_session = session
        if media:
            result = await adapter.send_document("peer", str(document), caption="caption")
        else:
            result = await adapter.send("peer", "a" * 50 + "\n\n" + "b" * 50)

    assert result.success
    assert len(tokens) >= 2 and tokens[0] == initial_token
    assert all(token == renewed_token for token in tokens[1:])
    assert adapter._token_store.get("acct", "peer") == renewed_token
    restored = wx.ContextTokenStore(str(tmp_path))
    restored.restore("acct")
    assert restored.get("acct", "peer") == renewed_token


@pytest.mark.asyncio
@pytest.mark.parametrize(("accept_tokenless", "rotate"), [(False, False), (True, False), (False, True)])
async def test_invalidated_token_stays_removed_after_restore_and_recovery_is_bounded(
    tmp_path, monkeypatch, accept_tokenless, rotate,
):
    adapter = connected_adapter(tmp_path)
    await adapter._token_store.set("acct", "peer", "old")
    await adapter._token_store.set("acct", "other-peer", "unrelated")
    tokens = []

    async def reply(*args, **kwargs):
        tokens.append(kwargs["context_token"])
        if rotate:
            assert len(tokens) <= 3, "fresh inbound tokens must not cause unbounded retries"
            await adapter._token_store.set("acct", "peer", f"new-{len(tokens)}")
        return {"ret": 0} if kwargs["context_token"] is None and accept_tokenless else {
            "ret": -2, "errmsg": "prepare failed"}

    monkeypatch.setattr(wx, "_send_message", reply)
    result = await adapter.send("peer", "short reply")

    assert result.success is accept_tokenless
    assert tokens == (["old", "new-1", "new-2"] if rotate else ["old", None])
    assert adapter._rate_limit_cooldown_remaining() == 0
    restored = wx.ContextTokenStore(str(tmp_path))
    restored.restore("acct")
    assert restored.get("acct", "peer") == ("new-3" if rotate else None)
    assert restored.get("acct", "other-peer") == "unrelated"
