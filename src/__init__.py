"""
CrewAI A2A Adapter

A powerful, flexible adapter that enables CrewAI to seamlessly integrate with
A2A (Agent-to-Agent) protocol compliant agents, providing tool discovery,
execution, and streaming capabilities.
"""

from .a2a_core import (
    A2ATool,
    A2ACallToolResult,
    A2ASession,
    ToolAdapter,
    A2AToolDiscovery,
    a2a_session,
)

from .crewai_adapter import (
    A2ACrewAIAdapter,
    CrewAIToolkit,
    create_pydantic_model_from_schema,
)

from .streaming import (
    StreamingCallback,
    SimpleStreamingCallback,
    StreamingCrew,
)

__version__ = "0.1.0"

__all__ = [
    # Core A2A components
    "A2ATool",
    "A2ACallToolResult",
    "A2ASession",
    "ToolAdapter",
    "A2AToolDiscovery",
    "a2a_session",
    # CrewAI adapter
    "A2ACrewAIAdapter",
    "CrewAIToolkit",
    "create_pydantic_model_from_schema",
    # Streaming support
    "StreamingCallback",
    "SimpleStreamingCallback",
    "StreamingCrew",
]
