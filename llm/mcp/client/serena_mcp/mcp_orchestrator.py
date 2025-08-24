#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama + Serena MCP Orchestrator (config.json 분리, 경로 안전)
- LLM: Ollama (stream)
- Tool Server: Serena MCP (SSE)
- Loop: stream -> tool_call -> Serena -> feed tool result -> continue

필요 패키지:
  pip install -U ollama mcp

환경변수(선택):
  MCP_ORCH_CONFIG = /path/to/config.json

PyInstaller 빌드 시에도 config.json 옆에 두면 자동 인식됩니다.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
from mcp import Client as MCPClient
from mcp.transports.sse import SSEClientTransport


# --------------------------
# 경로/설정 유틸
# --------------------------
def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

def _exe_dir() -> Path:
    # PyInstaller 바이너리면 실행 파일 폴더, 아니면 스크립트 폴더
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def _xdg_config_home(app_name: str) -> Path:
    # XDG (Linux/macOS), Windows는 Roaming 동등 처리
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / app_name

def _candidate_config_paths(app_name: str = "mcp_orchestrator") -> List[Path]:
    env = os.environ.get("MCP_ORCH_CONFIG")
    paths = []
    if env:
        paths.append(Path(env).expanduser().resolve())
    # 실행파일/스크립트 옆
    paths.append((_exe_dir() / "config.json").resolve())
    # 현재 작업 디렉토리
    paths.append((Path.cwd() / "config.json").resolve())
    # XDG
    paths.append((_xdg_config_home(app_name) / "config.json").resolve())
    return paths

def load_config(app_name: str = "mcp_orchestrator") -> Dict[str, Any]:
    last_err = None
    for p in _candidate_config_paths(app_name):
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # 상대경로는 config 파일이 있는 디렉토리 기준으로 보정
                cfg = _normalize_config_paths(cfg, p.parent)
                _validate_and_fill_defaults(cfg)
                print(f"[config] loaded: {p}")
                return cfg
        except Exception as e:
            last_err = e
    raise FileNotFoundError(
        f"config.json을 찾지 못했거나 파싱 실패했습니다. 탐색 경로: "
        + ", ".join(str(p) for p in _candidate_config_paths(app_name))
        + (f" | 마지막 오류: {last_err}" if last_err else "")
    )

def _normalize_config_paths(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    # serena.project_path 등 경로 키 보정
    serena = cfg.setdefault("serena", {})
    proj = serena.get("project_path")
    if isinstance(proj, str) and proj.strip():
        pp = Path(proj)
        if not pp.is_absolute():
            serena["project_path"] = str((base_dir / pp).resolve())
    # XDG app 이름 기본값
    runtime = cfg.setdefault("runtime", {})
    if "xdg_app_name" not in runtime:
        runtime["xdg_app_name"] = "mcp_orchestrator"
    return cfg

def _validate_and_fill_defaults(cfg: Dict[str, Any]) -> None:
    # 기본값 채우기 + 필수 키 검증
    llm = cfg.setdefault("llm", {})
    llm.setdefault("model", "qwen2.5:7b-instruct")
    llm.setdefault("host", "http://localhost:11434")
    llm.setdefault("system_prompt", "You are a helpful assistant.")
    llm.setdefault("max_rounds", 6)

    serena = cfg.setdefault("serena", {})
    serena.setdefault("sse_url", "http://localhost:9121/sse")
    serena.setdefault("context", None)
    serena.setdefault("project_path", None)
    serena.setdefault("timeout_ms", 60000)

    runtime = cfg.setdefault("runtime", {})
    runtime.setdefault("user_prompt", "프로젝트에서 AUWActorBase 관련 코드를 찾아 요약해줘.")
    runtime.setdefault("log_level", "info")

    # 간단한 필수값 체크
    if not serena["sse_url"]:
        raise ValueError("serena.sse_url 누락")
    # project_path는 Serena 툴 세트에 따라 필수가 아닐 수 있으므로 강제하지 않음.


# --------------------------
# Serena MCP 래퍼
# --------------------------
class SerenaMCP:
    def __init__(self, sse_url: str, context: Optional[str], project_path: Optional[str], timeout_ms: int):
        headers = {}
        if context:
            headers["x-serena-context"] = context
        if project_path:
            headers["x-serena-project"] = project_path

        self.transport = SSEClientTransport(sse_url, headers=headers, timeout_ms=timeout_ms)
        self.client: Optional[MCPClient] = None

    async def connect(self):
        self.client = MCPClient(app_name="mcp-ollama-orchestrator", app_version="0.2.0", transport=self.transport)
        await self.client.connect()

    async def close(self):
        if self.client:
            await self.client.close()

    async def list_tools(self) -> List[Dict[str, Any]]:
        assert self.client is not None
        resp = await self.client.list_tools()
        return resp.get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        assert self.client is not None
        try:
            result = await self.client.call_tool(name=name, arguments=arguments)
            return {"status": "ok", "data": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# --------------------------
# 도구 스키마 변환
# --------------------------
def serena_tools_to_openai_tools(serena_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    openai_tools = []
    for t in serena_tools:
        name = t.get("name")
        desc = t.get("description", "")
        params_schema = t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}}
        if not name:
            # 이름이 없으면 건너뛴다.
            continue
        openai_tools.append({
            "type": "function",
            "function": {"name": name, "description": desc, "parameters": params_schema}
        })
    return openai_tools


# --------------------------
# Ollama 래퍼(스트리밍)
# --------------------------
class OllamaLLM:
    def __init__(self, model: str, host: Optional[str], system_prompt: Optional[str]):
        self.model = model
        if host:
            # 전역 클라이언트 Host 바인딩
            ollama.Client(host=host)
        self.system_prompt = system_prompt

    def stream_chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
        # system 프롬프트를 맨 앞에 삽입
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(messages)

        return ollama.chat(model=self.model, messages=msgs, tools=tools, stream=True)


# --------------------------
# Tool-call 추출/결과 메시지
# --------------------------
def extract_tool_calls_from_final(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    return message.get("tool_calls") or []

def make_tool_result_messages(tool_call_id: str, tool_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(tool_result, ensure_ascii=False)
    }]


# --------------------------
# 오케스트레이터 루프
# --------------------------
async def orchestrate_with_config(cfg: Dict[str, Any]) -> None:
    serena = SerenaMCP(
        sse_url=cfg["serena"]["sse_url"],
        context=cfg["serena"]["context"],
        project_path=cfg["serena"]["project_path"],
        timeout_ms=int(cfg["serena"]["timeout_ms"]),
    )
    await serena.connect()
    try:
        serena_tool_list = await serena.list_tools()
        tool_specs = serena_tools_to_openai_tools(serena_tool_list)

        llm = OllamaLLM(
            model=cfg["llm"]["model"],
            host=cfg["llm"]["host"],
            system_prompt=cfg["llm"]["system_prompt"],
        )

        messages: List[Dict[str, Any]] = [{"role": "user", "content": cfg["runtime"]["user_prompt"]}]
        max_rounds = int(cfg["llm"]["max_rounds"])

        round_idx = 0
        while round_idx < max_rounds:
            round_idx += 1
            print(f"\n=== Round {round_idx} ===\n")

            partial_text: List[str] = []
            last_msg_obj: Optional[Dict[str, Any]] = None

            print("--- LLM streaming start ---\n")
            for chunk in llm.stream_chat(messages, tools=tool_specs):
                msg = chunk.get("message")
                if msg and "content" in msg and msg["content"]:
                    delta = msg["content"]
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    partial_text.append(delta)
                    last_msg_obj = msg
                if chunk.get("done"):
                    break
            print("\n--- LLM streaming end ---\n")

            if not last_msg_obj:
                print("LLM 응답이 비어 종료합니다.")
                break

            tool_calls = extract_tool_calls_from_final(last_msg_obj)
            # 어시스턴트 응답 턴 추가(문맥 유지)
            messages.append({
                "role": "assistant",
                "content": "".join(partial_text) if partial_text else (last_msg_obj.get("content") or "")
            })

            if not tool_calls:
                print("툴 호출 없음. 종료.")
                break

            # 여러 툴 동시 호출 처리(단순 직렬; 필요 시 asyncio.gather로 병렬화)
            appended = []
            for call in tool_calls:
                call_id = call.get("id") or f"call_{int(time.time()*1000)}"
                fn = call.get("function") or {}
                name = fn.get("name")
                args_str = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except Exception:
                    args = {}

                print(f"[Tool] {name}({args}) 호출 → Serena ...")
                tool_result = await serena.call_tool(name=name, arguments=args)
                appended.extend(make_tool_result_messages(call_id, tool_result))

            # 툴 결과 턴 주입 후 다음 라운드로
            messages.extend(appended)

        print("\n=== Done ===")

    finally:
        await serena.close()


# --------------------------
# 엔트리포인트
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Ollama + Serena MCP Orchestrator (config-driven)")
    parser.add_argument("--config", help="기본 탐색 규칙을 무시하고 이 경로의 config.json 사용")
    args = parser.parse_args()

    # 1) config 경로 강제 지정 시, 환경변수보다 우선
    if args.config:
        os.environ["MCP_ORCH_CONFIG"] = args.config

    cfg = load_config()  # 경로 우선순위/상대경로 보정/기본값 처리

    # 필요 시 XDG 위치에 기본 config 폴더 생성(로그/캐시 등 보관)
    xdg_dir = _xdg_config_home(cfg["runtime"]["xdg_app_name"])
    xdg_dir.mkdir(parents=True, exist_ok=True)

    import asyncio
    try:
        asyncio.run(orchestrate_with_config(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
