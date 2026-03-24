# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenAI Agents SDK adapter."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.testclient import TestClient

from pgns_agent import DEFAULT_HANDLER_NAME, Adapter, AgentServer
from pgns_agent_openai import OpenAIAgentsAdapter

# ---------------------------------------------------------------------------
# Helpers — mock OpenAI Agents SDK objects
# ---------------------------------------------------------------------------


def _make_usage(
    *, input_tokens: int = 50, output_tokens: int = 20, total_tokens: int = 70, requests: int = 1
) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = total_tokens
    usage.requests = requests
    return usage


def _make_context_wrapper(usage: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.usage = usage or _make_usage()
    return ctx


def _make_agent(name: str = "test-agent", model: str = "gpt-4o") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.model = model
    return agent


def _make_run_result(
    *,
    final_output: Any = "Hello, world!",
    agent: MagicMock | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    result = MagicMock()
    result.final_output = final_output
    result.last_agent = agent or _make_agent()
    result.context_wrapper = _make_context_wrapper(usage)
    return result


class _FakeTextDelta:
    """Mimics ``openai.types.responses.ResponseTextDeltaEvent``."""

    def __init__(self, delta: str) -> None:
        self.delta = delta


class _FakeStreamResult:
    """Mimics ``RunResultStreaming`` with async event iteration."""

    def __init__(
        self,
        deltas: list[str],
        final_output: str = "Hello, world!",
        agent: MagicMock | None = None,
        usage: MagicMock | None = None,
    ) -> None:
        self._deltas = deltas
        self.final_output = final_output
        self.last_agent = agent or _make_agent()
        self.context_wrapper = _make_context_wrapper(usage)

    async def stream_events(self) -> AsyncIterator[MagicMock]:
        for delta in self._deltas:
            event = MagicMock()
            event.type = "raw_response_event"
            event.data = _FakeTextDelta(delta)
            yield event


async def _collect(aiter: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect all chunks from an async iterator."""
    return [chunk async for chunk in aiter]


# ---------------------------------------------------------------------------
# Adapter base class contract
# ---------------------------------------------------------------------------


class TestAdapterContract:
    def test_is_adapter_subclass(self) -> None:
        agent = _make_agent()
        adapter = OpenAIAgentsAdapter(agent=agent)
        assert isinstance(adapter, Adapter)

    def test_default_stream_false(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        assert adapter.stream is False

    def test_default_max_turns(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        assert adapter.max_turns == 10


# ---------------------------------------------------------------------------
# Sync mode (stream=False returns a single dict)
# ---------------------------------------------------------------------------


class TestSyncMode:
    async def test_basic_run(self) -> None:
        mock_agent = _make_agent()
        adapter = OpenAIAgentsAdapter(agent=mock_agent)
        run_result = _make_run_result(final_output="Hi there!")

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            result = await adapter.handle({"prompt": "Hello"})

        assert result["output"] == "Hi there!"
        assert result["metadata"]["agent"] == "test-agent"
        assert result["metadata"]["model"] == "gpt-4o"
        assert result["metadata"]["usage"]["total_tokens"] == 70

    async def test_returns_dict_not_async_iterator(self) -> None:
        """Sync mode must return a plain dict, not an async generator."""
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            result = await adapter.handle({"prompt": "hello"})

        assert isinstance(result, dict)

    async def test_extracts_prompt_key(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"prompt": "test prompt"})
            MockRunner.run.assert_called_once()
            call_args = MockRunner.run.call_args
            assert call_args[0][1] == "test prompt"

    async def test_extracts_message_key(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"message": "test message"})
            assert MockRunner.run.call_args[0][1] == "test message"

    async def test_extracts_text_key(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"text": "test text"})
            assert MockRunner.run.call_args[0][1] == "test text"

    async def test_fallback_json_dumps(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"foo": "bar", "baz": 1})
            prompt = MockRunner.run.call_args[0][1]
            assert '"foo"' in prompt
            assert '"bar"' in prompt

    async def test_structured_output_pydantic(self) -> None:
        """Structured output with model_dump() (Pydantic-like)."""
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"answer": 42}
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result(final_output=mock_output)

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            result = await adapter.handle({"prompt": "what is 6*7?"})

        assert result["output"] == {"answer": 42}

    async def test_structured_output_dataclass(self) -> None:
        """Structured output with a plain dataclass."""

        @dataclasses.dataclass
        class Answer:
            value: int

        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result(final_output=Answer(value=42))

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            result = await adapter.handle({"prompt": "what is 6*7?"})

        assert result["output"] == {"value": 42}

    async def test_structured_output_plain_dict(self) -> None:
        """Plain dict output passes through unchanged."""
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result(final_output={"raw": "value"})

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            result = await adapter.handle({"prompt": "hello"})

        assert result["output"] == {"raw": "value"}

    async def test_max_turns_forwarded(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent(), max_turns=5)
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"prompt": "hello"})
            assert MockRunner.run.call_args.kwargs["max_turns"] == 5

    async def test_run_config_forwarded(self) -> None:
        run_config = MagicMock()
        adapter = OpenAIAgentsAdapter(agent=_make_agent(), run_config=run_config)
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"prompt": "hello"})
            assert MockRunner.run.call_args.kwargs["run_config"] is run_config

    async def test_model_none(self) -> None:
        """Agent with model=None should report None in metadata."""
        agent = _make_agent()
        agent.model = None
        adapter = OpenAIAgentsAdapter(agent=agent)
        run_result = _make_run_result(agent=agent)

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            result = await adapter.handle({"prompt": "hello"})

        assert result["metadata"]["model"] is None


# ---------------------------------------------------------------------------
# Streaming mode
# ---------------------------------------------------------------------------


class TestStreamingMode:
    async def test_yields_deltas_and_final(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent(), stream=True)
        stream_result = _FakeStreamResult(
            deltas=["Hello", ", ", "world!"],
            final_output="Hello, world!",
        )

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            with patch("pgns_agent_openai._adapter.ResponseTextDeltaEvent", _FakeTextDelta):
                MockRunner.run_streamed = MagicMock(return_value=stream_result)
                result = await adapter.handle({"prompt": "hi"})
                chunks = await _collect(result)

        # 3 deltas + 1 final
        assert len(chunks) == 4
        assert chunks[0] == {"delta": "Hello"}
        assert chunks[1] == {"delta": ", "}
        assert chunks[2] == {"delta": "world!"}
        assert chunks[3]["output"] == "Hello, world!"
        assert chunks[3]["metadata"]["usage"]["total_tokens"] == 70

    async def test_empty_stream(self) -> None:
        adapter = OpenAIAgentsAdapter(agent=_make_agent(), stream=True)
        stream_result = _FakeStreamResult(deltas=[], final_output="done")

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            with patch("pgns_agent_openai._adapter.ResponseTextDeltaEvent", _FakeTextDelta):
                MockRunner.run_streamed = MagicMock(return_value=stream_result)
                result = await adapter.handle({"prompt": "hi"})
                chunks = await _collect(result)

        # Only the final chunk
        assert len(chunks) == 1
        assert chunks[0]["output"] == "done"

    async def test_non_text_events_filtered(self) -> None:
        """Non-text-delta events (tool calls, handoffs) are dropped."""
        adapter = OpenAIAgentsAdapter(agent=_make_agent(), stream=True)
        stream_result = _FakeStreamResult(
            deltas=["Hi"],
            final_output="Hi",
        )

        # Inject a non-matching event between the text delta and final
        original_stream = stream_result.stream_events

        async def mixed_stream() -> AsyncIterator[MagicMock]:
            async for event in original_stream():
                yield event
            # Emit a tool-call event that should be filtered out
            tool_event = MagicMock()
            tool_event.type = "agent_updated"
            tool_event.data = MagicMock()
            yield tool_event

        stream_result.stream_events = mixed_stream

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            with patch("pgns_agent_openai._adapter.ResponseTextDeltaEvent", _FakeTextDelta):
                MockRunner.run_streamed = MagicMock(return_value=stream_result)
                result = await adapter.handle({"prompt": "hi"})
                chunks = await _collect(result)

        # 1 text delta + 1 final (agent_updated event filtered out)
        assert len(chunks) == 2
        assert chunks[0] == {"delta": "Hi"}
        assert "output" in chunks[1]


# ---------------------------------------------------------------------------
# Integration with AgentServer.use()
# ---------------------------------------------------------------------------


class TestAgentServerIntegration:
    def test_registers_default_handler(self) -> None:
        agent_server = AgentServer("a", "b")
        agent_server.use(OpenAIAgentsAdapter(agent=_make_agent()))
        assert DEFAULT_HANDLER_NAME in agent_server.handlers

    def test_registers_named_skill(self) -> None:
        agent_server = AgentServer("a", "b")
        agent_server.use(OpenAIAgentsAdapter(agent=_make_agent()), skill="openai")
        assert "openai" in agent_server.handlers

    def test_sync_adapter_via_http(self) -> None:
        agent_server = AgentServer("a", "b")
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        agent_server.use(adapter)
        run_result = _make_run_result(final_output="server response")

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            client = TestClient(agent_server.app())
            resp = client.post("/", json={"id": "t1", "input": {"prompt": "hello"}})

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "t1"
        assert data["result"]["output"] == "server response"
        assert data["result"]["metadata"]["agent"] == "test-agent"

    def test_streaming_adapter_via_http(self) -> None:
        agent_server = AgentServer("a", "b")
        adapter = OpenAIAgentsAdapter(agent=_make_agent(), stream=True)
        agent_server.use(adapter)
        stream_result = _FakeStreamResult(
            deltas=["Hi", "!"],
            final_output="Hi!",
        )

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            with patch("pgns_agent_openai._adapter.ResponseTextDeltaEvent", _FakeTextDelta):
                MockRunner.run_streamed = MagicMock(return_value=stream_result)
                client = TestClient(agent_server.app())
                resp = client.post("/", json={"id": "t1", "input": {"prompt": "hi"}})

        assert resp.status_code == 200
        # Streaming adapter returns the last chunk as the HTTP result
        result = resp.json()["result"]
        assert result["output"] == "Hi!"

    def test_adapter_exception_returns_500(self) -> None:
        agent_server = AgentServer("a", "b")
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        agent_server.use(adapter)

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(side_effect=RuntimeError("API error"))
            client = TestClient(agent_server.app(), raise_server_exceptions=False)
            resp = client.post("/", json={"id": "t1", "input": {"prompt": "fail"}})

        assert resp.status_code == 500

    def test_null_input_normalized(self) -> None:
        agent_server = AgentServer("a", "b")
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        agent_server.use(adapter)
        run_result = _make_run_result(final_output="ok")

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            client = TestClient(agent_server.app())
            resp = client.post("/", json={"id": "t1"})

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _extract_prompt edge cases
# ---------------------------------------------------------------------------


class TestExtractPrompt:
    async def test_non_string_priority_key_skipped(self) -> None:
        """When a priority key exists but is non-string, it should be skipped."""
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"prompt": 42, "message": "real prompt"})
            assert MockRunner.run.call_args[0][1] == "real prompt"

    async def test_non_string_all_keys_falls_back(self) -> None:
        """When all priority keys are non-string, falls back to json.dumps."""
        adapter = OpenAIAgentsAdapter(agent=_make_agent())
        run_result = _make_run_result()

        with patch("pgns_agent_openai._adapter.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=run_result)
            await adapter.handle({"prompt": 42})
            prompt = MockRunner.run.call_args[0][1]
            assert '"prompt"' in prompt
            assert "42" in prompt
