"""
Pytest configuration and fixtures for CrewAI A2A Adapter tests
"""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_a2a_tool():
    """Fixture providing a standard mock A2A tool"""
    from src.a2a_core import A2ATool

    return A2ATool(
        name="test_tool",
        description="A test tool for unit testing",
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Test input parameter"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of items",
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["input"]
        },
        output_schema={
            "type": "string",
            "description": "Tool output"
        }
    )


@pytest.fixture
def mock_server_params():
    """Fixture providing standard server parameters"""
    return {
        "url": "https://test-server.example.com",
        "headers": {
            "Authorization": "Bearer test-token-12345",
            "Content-Type": "application/json"
        }
    }


@pytest.fixture
def mock_agent_card():
    """Fixture providing a mock A2A agent card"""
    skill1 = MagicMock()
    skill1.id = "data_analysis"
    skill1.name = "data_analysis"
    skill1.description = "Analyze datasets and generate insights"

    skill2 = MagicMock()
    skill2.id = "text_processing"
    skill2.name = "text_processing"
    skill2.description = "Process and analyze text documents"

    card = MagicMock()
    card.skills = [skill1, skill2]
    card.name = "Test Agent"
    card.description = "A test agent for unit testing"

    return card


@pytest.fixture
def mock_crewai_agent():
    """Fixture providing a mock CrewAI agent"""
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.goal = "Complete test tasks"
    agent.backstory = "A helpful test agent"
    agent.tools = []
    agent.verbose = True

    return agent


@pytest.fixture
def mock_crewai_task():
    """Fixture providing a mock CrewAI task"""
    task = MagicMock()
    task.description = "Complete a test task"
    task.expected_output = "Task completion report"
    task.agent = None  # Will be set by tests

    return task


@pytest.fixture
def mock_crewai_crew(mock_crewai_agent, mock_crewai_task):
    """Fixture providing a mock CrewAI crew"""
    mock_crewai_task.agent = mock_crewai_agent

    crew = MagicMock()
    crew.agents = [mock_crewai_agent]
    crew.tasks = [mock_crewai_task]
    crew.manager_agent = None
    crew.process = "sequential"

    return crew


@pytest.fixture
def sample_json_schema():
    """Fixture providing a comprehensive JSON schema for testing"""
    return {
        "type": "object",
        "title": "ComplexToolInput",
        "description": "A complex tool input schema for testing",
        "properties": {
            "text": {
                "type": "string",
                "description": "Input text to process",
                "minLength": 1,
                "maxLength": 1000
            },
            "count": {
                "type": "integer",
                "description": "Number of iterations",
                "minimum": 1,
                "maximum": 100,
                "default": 10
            },
            "threshold": {
                "type": "number",
                "description": "Threshold value",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "enabled": {
                "type": "boolean",
                "description": "Whether the feature is enabled",
                "default": True
            },
            "options": {
                "type": "object",
                "description": "Additional options",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["json", "xml", "csv"],
                        "default": "json"
                    },
                    "compression": {
                        "type": "boolean",
                        "default": False
                    }
                }
            },
            "tags": {
                "type": "array",
                "description": "List of tags",
                "items": {
                    "type": "string"
                },
                "maxItems": 10
            }
        },
        "required": ["text", "count"],
        "additionalProperties": False
    }


@pytest.fixture
def event_loop():
    """Fixture providing a clean event loop for each test"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def configure_logging():
    """Auto-used fixture to configure logging for tests"""
    logging.getLogger("src").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


@pytest.fixture
def mock_httpx_client():
    """Fixture providing a mock httpx AsyncClient"""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {"status": "ok"}
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"result": "success"}

    return client


@pytest.fixture
def timeout_config():
    """Fixture providing timeout configuration for tests"""
    from datetime import timedelta

    return {
        "short": timedelta(seconds=1),
        "medium": timedelta(seconds=5),
        "long": timedelta(seconds=30)
    }


class MockA2ACallToolResult:
    """Mock implementation of A2ACallToolResult for testing"""

    def __init__(self, success=True, result=None, error=None):
        self.success = success
        self.result = result
        self.error = error


@pytest.fixture
def mock_successful_tool_result():
    """Fixture providing a successful A2A tool call result"""
    return MockA2ACallToolResult(
        success=True,
        result="Tool executed successfully with test data"
    )


@pytest.fixture
def mock_failed_tool_result():
    """Fixture providing a failed A2A tool call result"""
    return MockA2ACallToolResult(
        success=False,
        result=None,
        error="Tool execution failed: Invalid input parameter"
    )


@pytest.fixture
def multiple_server_configs():
    """Fixture providing multiple server configurations for testing"""
    return [
        {
            "url": "https://server1.example.com",
            "headers": {"Authorization": "Bearer token1"}
        },
        {
            "url": "https://server2.example.com",
            "headers": {"Authorization": "Bearer token2"}
        },
        {
            "url": "https://server3.example.com",
            # No headers - test unauthenticated server
        }
    ]


@pytest.fixture
def complex_a2a_tools():
    """Fixture providing multiple A2A tools with different schemas"""
    from src.a2a_core import A2ATool

    return [
        A2ATool(
            name="simple_tool",
            description="A simple tool with minimal schema",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                }
            }
        ),
        A2ATool(
            name="complex_tool",
            description="A complex tool with rich schema",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "values": {"type": "array", "items": {"type": "number"}},
                            "metadata": {"type": "object"}
                        }
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "format": {"type": "string", "enum": ["json", "csv"]},
                            "validate": {"type": "boolean"}
                        }
                    }
                },
                "required": ["data"]
            }
        ),
        A2ATool(
            name="no_schema_tool",
            description="A tool with empty schema",
            input_schema={}
        )
    ]


# Async helper fixtures

@pytest.fixture
async def async_mock_session():
    """Fixture providing an async mock A2A session"""
    from unittest.mock import AsyncMock
    from src.a2a_core import A2ACallToolResult

    session = AsyncMock()
    session.server_url = "https://mock-server.com"
    session.headers = {"Authorization": "Bearer mock-token"}
    session.tools = []

    # Default successful tool call
    session.call_tool.return_value = A2ACallToolResult(
        success=True,
        result="Mock tool execution result"
    )

    return session


# Performance testing fixtures

@pytest.fixture
def benchmark_tools(request):
    """Fixture for performance testing with configurable number of tools"""
    from src.a2a_core import A2ATool

    count = getattr(request, 'param', 10)  # Default to 10 tools
    tools = []

    for i in range(count):
        tool = A2ATool(
            name=f"benchmark_tool_{i}",
            description=f"Benchmark tool number {i}",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "iterations": {"type": "integer", "default": 1}
                }
            }
        )
        tools.append(tool)

    return tools


# Cleanup fixtures

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Auto-used fixture to perform cleanup after each test"""
    yield
    # Any cleanup code goes here
    # For example: clear global state, close connections, etc.
    pass