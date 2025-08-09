# mcp_client.py
"""
Refactored MCP JSON-RPC 2.0 client
- Supports Streamable-HTTP and plain HTTP
- Auto-detect with fallback
- Session handling for streamable-http (Mcp-Session-Id)
- Tolerant SSE/JSON parsing
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import aiohttp


@dataclass
class ClientConfig:
    server_url: str
    timeout: int
    transport_mode: str  # "auto" | "streamable-http" | "http"

    @staticmethod
    def from_config(path: Path) -> "ClientConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        c = raw.get("client", {})
        url = c.get("server_url", "http://127.0.0.1:4200/mcp/")
        # normalize: ensure trailing slash to avoid 307
        if not url.endswith("/"):
            url = url + "/"
        return ClientConfig(
            server_url=url,
            timeout=int(c.get("request_timeout_seconds", 15)),
            transport_mode=c.get("transport_mode", "auto"),
        )


class JsonRpcClient:
    def __init__(self, server_url: str, timeout: int = 15, transport_mode: str = "auto"):
        self.server_url = server_url
        self.timeout = timeout
        self.transport_mode = transport_mode  # auto | streamable-http | http
        self.session_id: Optional[str] = None
        self.negotiated_mode: Optional[str] = None  # resolved after initialize

    # ---------------------------- low-level helpers ----------------------------

    @staticmethod
    async def _read_first_json_event_or_none(resp: aiohttp.ClientResponse) -> Optional[Dict[str, Any]]:
        """Return JSON if Content-Type is application/json; or first SSE 'data:' JSON event; else None."""
        ctype = (resp.headers.get("Content-Type") or "").lower()

        # JSON 응답
        if ctype.startswith("application/json"):
            try:
                return await resp.json()
            except Exception:
                # 혹시 헤더만 json이고 바디가 비어있을 수도 있음
                text = await resp.text()
                try:
                    return json.loads(text)
                except Exception:
                    return None

        # SSE 응답
        if ctype.startswith("text/event-stream"):
            buffer: list[str] = []
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")

                if not line:
                    # 이벤트 경계: data: 라인만 취합
                    data_lines = [l[5:].lstrip() for l in buffer if l.startswith("data:")]
                    buffer = []
                    if not data_lines:
                        continue
                    payload = "\n".join(data_lines).strip()
                    if not payload:
                        continue
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        # JSON이 아니면 다음 이벤트 대기
                        continue
                else:
                    if line.startswith(":"):
                        # SSE 주석 라인은 무시
                        continue
                    # event: retry: id: 등은 버퍼에 쌓되, 경계에서 data:만 추출함
                    buffer.append(line)

            # EOF: 남은 버퍼 플러시
            if buffer:
                data_lines = [l[5:].lstrip() for l in buffer if l.startswith("data:")]
                payload = "\n".join(data_lines).strip()
                if payload:
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        return None
            return None

        # 헤더가 부정확한 서버 대응: 텍스트로 읽어 JSON 시도
        try:
            text = await resp.text()
            return json.loads(text)
        except Exception:
            return None

    async def _post(
        self,
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
        mode: str,
    ) -> Tuple[Optional[Dict[str, Any]], aiohttp.ClientResponse]:
        """Send POST with proper headers for given mode and return (parsed_body_or_none, resp)."""
        if mode not in ("streamable-http", "http"):
            raise ValueError(f"Unknown mode: {mode}")

        # ★ 모드와 무관하게 Accept는 둘 다 명시 (서버 협상 요구 충족)
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if mode == "streamable-http" and self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        r = await session.post(self.server_url, json=payload, headers=headers)
        # streamable 시도에서 406이 오면 호출자가 폴백 시도
        if r.status == 406 and mode == "streamable-http":
            body = await r.text()
            raise RuntimeError(f"406 Not Acceptable in streamable-http: {body}")
        r.raise_for_status()
        parsed = await self._read_first_json_event_or_none(r)
        return parsed, r

    # ---------------------------- initialization ----------------------------

    async def initialize(self, protocol_version: str = "2025-03-26") -> Dict[str, Any]:
        """Initialize session. In 'auto' mode: try streamable-http first, then fallback to http."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            req = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "protocolVersion": protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "kojeom-client", "version": "1.0.0"},
                },
            }

            # trial order
            trial_order = ["streamable-http", "http"] if self.transport_mode == "auto" else [self.transport_mode]

            last_err: Optional[Exception] = None
            for mode in trial_order:
                try:
                    parsed, resp = await self._post(s, req, mode=mode)

                    if mode == "streamable-http":
                        sid = resp.headers.get("mcp-session-id")
                        if sid:
                            # 세션이 있으면 바디가 없어도 성공으로 간주
                            self.session_id = sid
                            self.negotiated_mode = mode
                            return parsed or {"info": "initialize: SSE body empty, session established"}
                        else:
                            # 세션이 반드시 필요
                            raise RuntimeError("Missing Mcp-Session-Id in initialize response (streamable-http).")

                    # http 모드: 세션 없음, 바디 없을 수도 있음
                    self.negotiated_mode = mode
                    return parsed or {"info": "initialize: empty body (http mode)"}

                except Exception as e:
                    last_err = e
                    if mode == "streamable-http" and self.transport_mode == "auto":
                        # http 폴백 시도
                        continue
                    break

            raise RuntimeError(f"initialize failed (mode={self.transport_mode}): {last_err}")

    async def initialized(self) -> Dict[str, Any]:
        """Send 'initialized' notification (recommended) if streamable-http negotiated."""
        if self.negotiated_mode != "streamable-http":
            return {"info": "skip initialized on http mode"}

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            req = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialized",
                "params": {},
            }
            parsed, _ = await self._post(s, req, mode="streamable-http")
            return parsed or {"ok": True}

    # ---------------------------- calls ----------------------------

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.negotiated_mode:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            req = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params or {},
            }
            parsed, _ = await self._post(s, req, mode=self.negotiated_mode)
            # JSON-RPC 에러는 서버가 error 필드로 내려줄 수 있으므로 여기서 체크해도 됨
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"JSON-RPC error: {parsed['error']}")
            return parsed or {"info": "empty body"}

    async def tool_call(self, name: str, **kwargs) -> Dict[str, Any]:
        return await self.call(name, kwargs)

    # ---------------------------- streaming advanced (optional) ----------------------------

    async def call_stream(self, method: str, params: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """If server produces multiple SSE events for a call, stream them incrementally.
        Only supported in streamable-http mode. Yields each event JSON (parsed)."""
        if self.negotiated_mode != "streamable-http":
            raise RuntimeError("Streaming only supported in streamable-http mode.")

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            req = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params or {},
            }
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Mcp-Session-Id": self.session_id or "",
            }
            async with s.post(self.server_url, json=req, headers=headers) as resp:
                resp.raise_for_status()
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if not ctype.startswith("text/event-stream"):
                    # Fallback: 단발 JSON
                    try:
                        yield await resp.json()
                    except Exception:
                        text = await resp.text()
                        try:
                            yield json.loads(text)
                        except Exception:
                            yield {"info": "empty body"}
                    return

                buffer: list[str] = []
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    if not line:
                        data_lines = [l[5:].lstrip() for l in buffer if l.startswith("data:")]
                        buffer = []
                        if not data_lines:
                            continue
                        payload = "\n".join(data_lines).strip()
                        if not payload:
                            continue
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                    else:
                        if line.startswith(":"):
                            continue
                        buffer.append(line)

                # EOF: 남은 이벤트 한 번 더 처리
                if buffer:
                    data_lines = [l[5:].lstrip() for l in buffer if l.startswith("data:")]
                    payload = "\n".join(data_lines).strip()
                    if payload:
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            pass


async def _demo():
    cfg = ClientConfig.from_config(Path(__file__).with_name("config.json"))
    client = JsonRpcClient(cfg.server_url, cfg.timeout, cfg.transport_mode)

    # 1) negotiate
    init_res = await client.initialize()
    print("[initialize]", init_res, "| mode:", client.negotiated_mode, "| sid:", client.session_id)

    # 2) (recommended for streamable-http)
    if client.negotiated_mode == "streamable-http":
        print("[initialized]", await client.initialized())

    # 3) calls
    print("[ping]", await client.tool_call("ping"))
    print("[greet]", await client.tool_call("greet", name="코점스튜디오"))
    print("[find_products]", await client.tool_call("find_products", query="apple", category="bakery"))


if __name__ == "__main__":
    asyncio.run(_demo())