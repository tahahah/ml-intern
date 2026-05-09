"""
ml-intern ACP adapter — makes ml-intern speak the Agent Client Protocol
so Happy can drive it like any other ACP-compatible agent.

Wire protocol: JSON-RPC 2.0 over newline-delimited JSON on stdin/stdout.

Usage (after uv tool install -e .):
    happy acp ml-intern          # using KNOWN_ACP_AGENTS entry
    happy acp -- ml-intern-acp   # or directly

The adapter:
  - Implements the ACP Agent interface (initialize / newSession / prompt /
    setSessionMode / authenticate / cancel)
  - Translates Happy prompt turns → ml-intern Submission(OpType.USER_INPUT)
  - Streams ml-intern events → ACP sessionUpdate notifications
  - Handles approval_required ↔ ACP requestPermission (one request per tool)
  - Maps session/cancel → OpType.INTERRUPT
  - Credentials (HF_TOKEN, model keys) stay on the machine; Happy never sees them
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from agent.config import load_config
from agent.core.agent_loop import submission_loop
from agent.core.hf_tokens import resolve_hf_token
from agent.core.local_models import is_local_model_id
from agent.core.session import OpType
from agent.core.tools import ToolRouter
from agent.main import Operation, Submission

logger = logging.getLogger(__name__)

CLI_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "cli_agent_config.json"

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 wire helpers
# ---------------------------------------------------------------------------

def _rpc_response(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _rpc_error(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _rpc_request(id_: str, method: str, params: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


def _rpc_notification(method: str, params: Any) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params}


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

class _SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        # Queue feeding ml-intern (Submission objects)
        self.submission_queue: asyncio.Queue = asyncio.Queue()
        # Queue reading ml-intern events (Event objects)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        # Resolved when current prompt turn ends (value = stopReason string)
        self.turn_future: asyncio.Future | None = None
        # Running counter for submission IDs
        self._sub_id: int = 0
        # Background tasks
        self.agent_task: asyncio.Task | None = None
        self.event_task: asyncio.Task | None = None
        self.started: bool = False

    def next_sub_id(self) -> str:
        self._sub_id += 1
        return f"sub_{self._sub_id}"

    def make_submission(self, op_type: OpType, data: dict | None = None) -> Submission:
        return Submission(
            id=self.next_sub_id(),
            operation=Operation(op_type=op_type, data=data),
        )


# ---------------------------------------------------------------------------
# ACP ↔ ml-intern bridge
# ---------------------------------------------------------------------------

class MlInternAcpServer:
    """
    Runs as the subprocess that Happy spawns via `happy acp ml-intern`.

    Reads ACP JSON-RPC messages from stdin, dispatches, writes responses
    and notifications to stdout.  All ml-intern credentials stay local.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        # Pending outgoing requests (id → Future)
        self._pending: dict[str, asyncio.Future] = {}
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _write(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        async with self._write_lock:
            self._writer.write(line.encode())
            await self._writer.drain()

    async def _notify(self, method: str, params: Any) -> None:
        await self._write(_rpc_notification(method, params))

    async def _session_update(self, session_id: str, update: dict) -> None:
        await self._notify("session/update", {"sessionId": session_id, "update": update})

    async def _outgoing_request(self, method: str, params: Any) -> Any:
        """Send a request to the client (Happy) and await its response."""
        req_id = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        await self._write(_rpc_request(req_id, method, params))
        return await fut

    # ------------------------------------------------------------------
    # ACP method handlers (called from main dispatch loop)
    # ------------------------------------------------------------------

    async def _on_initialize(self, id_: Any, _params: dict) -> None:
        await self._write(_rpc_response(id_, {
            "protocolVersion": "0.14",
            "agentCapabilities": {"loadSession": False},
            "displayName": "ML Intern",
        }))

    async def _on_new_session(self, id_: Any, _params: dict) -> None:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = _SessionState(session_id)
        await self._write(_rpc_response(id_, {"sessionId": session_id}))

    async def _on_authenticate(self, id_: Any, _params: dict) -> None:
        await self._write(_rpc_response(id_, {}))

    async def _on_set_session_mode(self, id_: Any, _params: dict) -> None:
        await self._write(_rpc_response(id_, {}))

    async def _on_prompt(self, id_: Any, params: dict) -> None:
        session_id = params.get("sessionId", "")
        state = self._sessions.get(session_id)
        if not state:
            await self._write(_rpc_error(id_, -32600, f"Unknown session: {session_id}"))
            return

        # Extract text from ACP content blocks
        text = _extract_text(params.get("content") or [])

        # Lazily start the ml-intern agent loop on first prompt
        if not state.started:
            state.started = True
            asyncio.create_task(self._start_agent(state))
            # Give the agent loop a moment to spin up
            await asyncio.sleep(0.1)

        # Enqueue user input
        await state.submission_queue.put(
            state.make_submission(OpType.USER_INPUT, {"text": text})
        )

        # Await turn completion
        loop = asyncio.get_event_loop()
        state.turn_future = loop.create_future()
        try:
            stop_reason = await state.turn_future
        except asyncio.CancelledError:
            stop_reason = "cancelled"
        finally:
            state.turn_future = None

        await self._write(_rpc_response(id_, {"stopReason": stop_reason}))

    async def _on_cancel(self, id_: Any, params: dict) -> None:
        session_id = params.get("sessionId", "")
        state = self._sessions.get(session_id)
        if state:
            await state.submission_queue.put(
                state.make_submission(OpType.INTERRUPT)
            )
            if state.turn_future and not state.turn_future.done():
                state.turn_future.set_result("cancelled")
        if id_ is not None:
            await self._write(_rpc_response(id_, {}))

    # ------------------------------------------------------------------
    # ml-intern agent lifecycle
    # ------------------------------------------------------------------

    async def _start_agent(self, state: _SessionState) -> None:
        """Load config, create ToolRouter, start submission_loop and event consumer."""
        try:
            config = load_config(CLI_CONFIG_PATH, include_user_defaults=True)
        except Exception as e:
            logger.error("Failed to load ml-intern config: %s", e)
            return

        hf_token = resolve_hf_token()
        if not hf_token and not is_local_model_id(config.model_name):
            logger.warning("No HF token. Set HF_TOKEN or run `huggingface-cli login`.")

        tool_router = ToolRouter(config=config, hf_token=hf_token)
        await tool_router.initialize()

        state.event_task = asyncio.create_task(
            self._consume_events(state)
        )

        state.agent_task = asyncio.create_task(
            submission_loop(
                submission_queue=state.submission_queue,
                event_queue=state.event_queue,
                config=config,
                tool_router=tool_router,
                hf_token=hf_token,
                local_mode=False,
                stream=True,
            )
        )

        try:
            await state.agent_task
        except Exception as e:
            logger.error("ml-intern agent loop error: %s", e)
        finally:
            # Cancel event consumer when agent exits
            if state.event_task and not state.event_task.done():
                state.event_task.cancel()
            if state.turn_future and not state.turn_future.done():
                state.turn_future.set_result("error")

    # ------------------------------------------------------------------
    # Event → ACP translation
    # ------------------------------------------------------------------

    async def _consume_events(self, state: _SessionState) -> None:
        """Read events from ml-intern event_queue and translate to ACP protocol."""
        while True:
            try:
                event = await asyncio.wait_for(state.event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if state.agent_task and state.agent_task.done():
                    break
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Event queue error: %s", e)
                break

            await self._translate_event(state, event)

            if event.event_type == "session_terminated":
                break

    async def _translate_event(self, state: _SessionState, event: Any) -> None:
        session_id = state.session_id
        et = event.event_type
        data = event.data or {}

        # --- Text streaming ---
        if et == "assistant_chunk":
            text = data.get("content", "")
            if text:
                await self._session_update(session_id, {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                })

        elif et == "assistant_message":
            text = data.get("content", "")
            if text:
                await self._session_update(session_id, {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                })

        # --- Tool calls ---
        elif et == "tool_call":
            tool = data.get("tool", "unknown")
            call_id = data.get("tool_call_id", str(uuid.uuid4()))
            raw_input = data.get("arguments", {})
            await self._session_update(session_id, {
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": tool,
                "kind": "run",
                "status": "pending",
                "rawInput": raw_input if isinstance(raw_input, dict) else {},
            })

        elif et == "tool_output":
            call_id = data.get("tool_call_id", "")
            output = data.get("output", "")
            is_error = data.get("is_error", False)
            await self._session_update(session_id, {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "error" if is_error else "completed",
                "content": [{"type": "content",
                              "content": {"type": "text", "text": str(output)}}],
                "rawOutput": {"output": output},
            })

        elif et == "tool_log":
            msg = data.get("message", "")
            if msg:
                await self._session_update(session_id, {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": f"[log] {msg}\n"},
                })

        # --- Approvals ---
        elif et == "approval_required":
            await self._handle_approvals(state, data)

        # --- Turn lifecycle ---
        elif et == "turn_complete":
            if state.turn_future and not state.turn_future.done():
                state.turn_future.set_result("end_turn")

        elif et == "interrupted":
            if state.turn_future and not state.turn_future.done():
                state.turn_future.set_result("cancelled")

        elif et == "error":
            err_msg = data.get("error", "Unknown error")
            await self._session_update(session_id, {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"[error] {err_msg}\n"},
            })
            if state.turn_future and not state.turn_future.done():
                state.turn_future.set_result("error")

        # Silently absorb events with no ACP mapping
        # (compacted, processing, assistant_stream_end, tool_state_change,
        #  resume_complete, undo_complete, session_terminated)

    async def _handle_approvals(self, state: _SessionState, data: dict) -> None:
        """
        One requestPermission per tool.  Collect all responses, then submit
        a single EXEC_APPROVAL back to ml-intern.
        """
        session_id = state.session_id
        tools_data: list[dict] = data.get("tools", [])
        if not tools_data:
            return

        approvals: list[dict] = []
        for tool_info in tools_data:
            tool_name = tool_info.get("tool", "tool")
            call_id = tool_info.get("tool_call_id", str(uuid.uuid4()))
            raw_input = tool_info.get("arguments", {})

            try:
                resp = await self._outgoing_request(
                    "client/requestPermission",
                    {
                        "sessionId": session_id,
                        "toolCall": {
                            "toolCallId": call_id,
                            "title": tool_name,
                            "kind": "run",
                            "status": "pending",
                            "rawInput": raw_input if isinstance(raw_input, dict) else {},
                        },
                        "options": [
                            {"kind": "allow_once", "name": "Allow", "optionId": "allow"},
                            {"kind": "reject_once", "name": "Deny",  "optionId": "deny"},
                        ],
                    },
                )
                outcome = (resp or {}).get("outcome", {})
                approved = (
                    isinstance(outcome, dict) and outcome.get("optionId") == "allow"
                )
            except Exception as e:
                logger.warning("Permission request failed for %s: %s", tool_name, e)
                approved = False

            approvals.append({
                "tool_call_id": call_id,
                "approved": approved,
                "feedback": None,
            })

        await state.submission_queue.put(
            state.make_submission(OpType.EXEC_APPROVAL, {"approvals": approvals})
        )

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: dict) -> None:
        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        # Response to one of our outgoing requests (e.g. requestPermission)
        if "result" in msg and msg_id in self._pending:
            fut = self._pending.pop(msg_id)
            if not fut.done():
                fut.set_result(msg["result"])
            return
        if "error" in msg and msg_id in self._pending:
            fut = self._pending.pop(msg_id)
            if not fut.done():
                fut.set_exception(Exception(
                    msg["error"].get("message", "RPC error")
                ))
            return

        # Method routing
        if method == "initialize":
            await self._on_initialize(msg_id, params)
        elif method == "session/new":
            await self._on_new_session(msg_id, params)
        elif method == "authenticate":
            await self._on_authenticate(msg_id, params)
        elif method in ("session/set_mode", "session/set_config_option"):
            await self._on_set_session_mode(msg_id, params)
        elif method == "session/prompt":
            await self._on_prompt(msg_id, params)
        elif method == "session/cancel":
            await self._on_cancel(msg_id, params)
        else:
            if msg_id is not None:
                await self._write(_rpc_error(msg_id, -32601, f"Method not found: {method}"))

    # ------------------------------------------------------------------
    # Stdio server loop
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        loop = asyncio.get_event_loop()

        # Async stdin
        reader = asyncio.StreamReader()
        proto = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: proto, sys.stdin.buffer)

        # Async stdout
        w_transport, w_proto = await loop.connect_write_pipe(
            asyncio.BaseProtocol, sys.stdout.buffer
        )
        self._writer = asyncio.StreamWriter(w_transport, w_proto, None, loop)

        while True:
            try:
                line = await reader.readline()
            except Exception:
                break
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped)
            except json.JSONDecodeError as e:
                logger.warning("Bad JSON from client: %s", e)
                continue
            asyncio.create_task(self._dispatch(msg))

        # Clean up
        pending_tasks = [
            state.agent_task
            for state in self._sessions.values()
            if state.agent_task and not state.agent_task.done()
        ]
        if pending_tasks:
            for t in pending_tasks:
                t.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content: list) -> str:
    """Extract plain text from ACP content blocks."""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif "text" in block:
                parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cli() -> None:
    """
    Registered as ml-intern-acp in pyproject.toml.
    Called by Happy as the ACP subprocess.
    """
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    server = MlInternAcpServer()
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass
