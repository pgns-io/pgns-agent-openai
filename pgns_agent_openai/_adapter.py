# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Agents SDK adapter for pgns-agent."""

from __future__ import annotations

__all__ = ["OpenAIAgentsAdapter"]

import dataclasses
import json
from collections.abc import AsyncIterator
from typing import Any

from agents import Agent, RunConfig, Runner
from openai.types.responses import ResponseTextDeltaEvent

from pgns_agent import Adapter


def _usage_dict(usage: Any) -> dict[str, Any]:
    """Extract token-usage metadata from a Usage dataclass."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "requests": usage.requests,
    }


def _agent_model(agent: Agent[Any]) -> str | None:
    """Best-effort extraction of the model name from an Agent."""
    model = agent.model
    if model is None:
        return None
    if isinstance(model, str):
        return model
    # Model protocol objects expose a name or similar — fall back gracefully.
    return getattr(model, "model", None) or str(model)


def _build_result(result: Any) -> dict[str, Any]:
    """Build the result dict from a RunResult or RunResultStreaming."""
    output = result.final_output
    if not isinstance(output, str):
        if dataclasses.is_dataclass(output) and not isinstance(output, type):
            output = dataclasses.asdict(output)
        elif hasattr(output, "model_dump"):
            output = output.model_dump()
        # Other JSON-serialisable types (dict, list, primitives) pass through unchanged.
    return {
        "output": output,
        "metadata": {
            "agent": result.last_agent.name,
            "model": _agent_model(result.last_agent),
            "usage": _usage_dict(result.context_wrapper.usage),
        },
    }


@dataclasses.dataclass(slots=True)
class OpenAIAgentsAdapter(Adapter):
    """Thin adapter wrapping an OpenAI Agents SDK ``Agent`` into pgns-agent.

    Sync mode (default)::

        from agents import Agent
        from pgns_agent import AgentServer
        from pgns_agent_openai import OpenAIAgentsAdapter

        openai_agent = Agent(name="helper", instructions="You are helpful.")
        server = AgentServer("my-agent", "An agent powered by OpenAI")
        server.use(OpenAIAgentsAdapter(openai_agent))
        server.listen(3000)

    Streaming mode::

        server.use(OpenAIAgentsAdapter(openai_agent, stream=True))

    Args:
        agent: A configured ``agents.Agent`` instance.
        stream: When ``True``, use ``Runner.run_streamed()`` and yield text
            deltas as they arrive.  Defaults to ``False``.
        run_config: Optional ``RunConfig`` forwarded to the runner.
        max_turns: Maximum agent turns (default 10).
    """

    agent: Agent[Any]
    stream: bool = False
    run_config: RunConfig | None = None
    max_turns: int = 10

    async def handle(
        self, task_input: dict[str, Any]
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        prompt = _extract_prompt(task_input)

        if self.stream:
            return self._stream(prompt)

        result = await Runner.run(
            self.agent,
            prompt,
            run_config=self.run_config,
            max_turns=self.max_turns,
        )
        return _build_result(result)

    async def _stream(self, prompt: str) -> AsyncIterator[dict[str, Any]]:
        """Yield text-delta chunks followed by the final result."""
        streamed = Runner.run_streamed(
            self.agent,
            prompt,
            run_config=self.run_config,
            max_turns=self.max_turns,
        )
        async for event in streamed.stream_events():
            # Only text deltas are forwarded; tool-call and handoff events are
            # intentionally dropped — they are internal to the agent run.
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield {"delta": event.data.delta}
        yield _build_result(streamed)


def _extract_prompt(task_input: dict[str, Any]) -> str:
    """Normalise task_input into a prompt string for the OpenAI agent.

    Accepts:
    - ``{"prompt": "..."}``
    - ``{"message": "..."}``
    - ``{"text": "..."}``
    - ``{"input": "..."}``
    - Any dict — serialised to a compact string representation.
    """
    for key in ("prompt", "message", "text", "input"):
        if key in task_input and isinstance(task_input[key], str):
            return task_input[key]
    # Fallback: deterministic JSON serialisation of the full input dict.
    return json.dumps(task_input, default=str)
