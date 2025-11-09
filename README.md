# CrewAI A2A Adapter

Lightweight bridge that lets CrewAI crews call A2A protocol compliant agents. This proof-of-concept focuses on wiring the systems together with minimal ceremony.

## Highlights
- Discovers tools exposed by A2A agent cards and adapts them into CrewAI tools on the fly
- Runs async A2A calls from CrewAI's sync tool interface with streaming callbacks for progress
- Ships with pytest-covered core logic plus simple examples for sequential and streaming crews

## Installation
```bash
pip install -e .
```
Requirements: Python 3.11+ with valid credentials for your A2A services.

## Quick Start
```python
import asyncio
from crewai import Agent, Crew, Task
from src import CrewAIToolkit

async def run():
    toolkit = CrewAIToolkit()
    await toolkit.load_from_a2a_servers([
        {"url": "https://your-agent.example", "headers": {"Authorization": "Bearer TOKEN"}}
    ])

    researcher = Agent(
        role="Researcher",
        goal="Collect the latest findings",
        tools=toolkit.get_tools(),
        verbose=True,
    )

    task = Task(description="Summarise recent AI releases", agent=researcher)
    Crew(agents=[researcher], tasks=[task]).kickoff()

asyncio.run(run())
```

## Development Notes
- `pytest` exercises adapters, sessions, and streaming glue using heavy mocking
- `test_runner.py` runs lint/type checks for local use; nothing runs automatically here
- Examples under `examples/` illustrate sequential, streaming, and multi-server setups

## License
MIT
