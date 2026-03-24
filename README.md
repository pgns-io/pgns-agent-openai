# pgns-agent-openai

OpenAI Agents SDK adapter for [pgns-agent](https://pypi.org/project/pgns-agent/). Wrap any OpenAI agent in a production-ready A2A server with three lines of code.

## Installation

```bash
pip install pgns-agent-openai
```

## Quick Start

```python
from agents import Agent
from pgns_agent import AgentServer
from pgns_agent_openai import OpenAIAgentsAdapter

openai_agent = Agent(name="helper", instructions="You are a helpful assistant.")

server = AgentServer("my-agent", "An agent powered by OpenAI Agents SDK")
server.use(OpenAIAgentsAdapter(openai_agent))
server.listen(3000)
```

## Streaming

```python
server.use(OpenAIAgentsAdapter(openai_agent, stream=True))
```

## License

Apache-2.0
