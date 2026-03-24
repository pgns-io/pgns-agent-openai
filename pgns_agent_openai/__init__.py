# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""pgns-agent-openai — OpenAI Agents SDK adapter for pgns-agent."""

from pgns_agent_openai._adapter import OpenAIAgentsAdapter
from pgns_agent_openai._version import __version__

__all__ = [
    "OpenAIAgentsAdapter",
    "__version__",
]
