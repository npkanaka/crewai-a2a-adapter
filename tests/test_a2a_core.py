"""
Tests for A2A Core functionality
"""

import asyncio
import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from src.a2a_core import (
    A2ATool,
    A2ACallToolResult,
    A2ASession,
    A2AToolDiscovery,
    a2a_session,
)


class TestA2ATool:
    """Test A2ATool data class"""

    def test_a2a_tool_creation(self):
        tool = A2ATool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "string"}
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "properties" in tool.input_schema
        assert tool.output_schema == {"type": "string"}

    def test_a2a_tool_optional_output_schema(self):
        tool = A2ATool(
            name="simple_tool",
            description="Simple tool",
            input_schema={"type": "object"}
        )

        assert tool.output_schema is None


class TestA2ACallToolResult:
    """Test A2ACallToolResult data class"""

    def test_successful_result(self):
        result = A2ACallToolResult(
            success=True,
            result="Tool executed successfully"
        )

        assert result.success is True
        assert result.result == "Tool executed successfully"
        assert result.error is None

    def test_failed_result(self):
        result = A2ACallToolResult(
            success=False,
            result=None,
            error="Tool execution failed"
        )

        assert result.success is False
        assert result.result is None
        assert result.error == "Tool execution failed"


class TestA2ASession:
    """Test A2ASession class"""

    def setup_method(self):
        self.server_url = "https://test-server.com"
        self.headers = {"Authorization": "Bearer test-token"}
        self.session = A2ASession(
            server_url=self.server_url,
            headers=self.headers,
            timeout=timedelta(seconds=30)
        )

    def test_session_initialization(self):
        assert self.session.server_url == self.server_url
        assert self.session.headers == self.headers
        assert self.session.timeout == timedelta(seconds=30)
        assert self.session.tools == []
        assert self.session.httpx_client is None
        assert self.session.client is None

    @pytest.mark.asyncio
    async def test_session_connection_success(self):
        # Mock agent card with skills
        mock_skill = MagicMock()
        mock_skill.id = "test_skill"
        mock_skill.description = "Test skill description"

        mock_agent_card = MagicMock()
        mock_agent_card.skills = [mock_skill]

        with patch('src.a2a_core.httpx.AsyncClient') as mock_client_class, \
             patch('src.a2a_core.A2ACardResolver') as mock_resolver_class, \
             patch('src.a2a_core.A2AClient') as mock_a2a_client:

            # Setup mocks
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_resolver = AsyncMock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.get_agent_card.return_value = mock_agent_card

            mock_a2a = AsyncMock()
            mock_a2a_client.return_value = mock_a2a

            # Test connection
            await self.session.connect()

            # Verify setup
            assert self.session.agent_card == mock_agent_card
            assert len(self.session.tools) == 1
            assert self.session.tools[0].name == "test_skill"
            assert self.session.tools[0].description == "Test skill description"

    @pytest.mark.asyncio
    async def test_session_connection_failure(self):
        with patch('src.a2a_core.httpx.AsyncClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await self.session.connect()

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        result = await self.session.call_tool("test_tool")

        assert result.success is False
        assert result.error == "Client not connected"
        assert result.result is None

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        # Setup a connected session
        mock_client = AsyncMock()
        self.session.client = mock_client

        # Import required types
        from a2a.types import Message, Part, TextPart

        # Mock successful message response
        mock_text_part = TextPart(text="Tool result")
        mock_part = Part(root=mock_text_part)
        mock_message = Message(
            role="agent",
            parts=[mock_part],
            messageId="test-id",
            metadata={}
        )

        mock_response = MagicMock()
        mock_response.root.result = mock_message

        mock_client.send_message.return_value = mock_response

        result = await self.session.call_tool("test_tool", {"input": "test input"})

        assert result.success is True
        assert result.result == "Tool result"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self):
        # Create session with very short timeout
        short_session = A2ASession(
            server_url=self.server_url,
            timeout=timedelta(milliseconds=1)
        )
        mock_client = AsyncMock()
        short_session.client = mock_client

        # Mock slow response
        async def slow_send_message(*args, **kwargs):
            await asyncio.sleep(0.1)  # Sleep longer than timeout
            return MagicMock()

        mock_client.send_message = slow_send_message

        result = await short_session.call_tool("test_tool")

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        mock_client = AsyncMock()
        self.session.httpx_client = mock_client

        await self.session.disconnect()

        mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_a2a_session_context_manager():
    """Test the a2a_session context manager"""
    server_params = {
        "url": "https://test-server.com",
        "headers": {"Authorization": "Bearer test"}
    }

    with patch('src.a2a_core.A2ASession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session.tools = [
            A2ATool("tool1", "Test tool 1", {"type": "object"}),
            A2ATool("tool2", "Test tool 2", {"type": "object"})
        ]
        mock_session_class.return_value = mock_session

        async with a2a_session(server_params) as (session, tools):
            assert session == mock_session
            assert len(tools) == 2
            assert tools[0].name == "tool1"
            assert tools[1].name == "tool2"

        # Verify cleanup was called
        mock_session.connect.assert_called_once()
        mock_session.disconnect.assert_called_once()


class TestA2AToolDiscovery:
    """Test A2AToolDiscovery class"""

    def setup_method(self):
        self.server_params = [
            {"url": "https://server1.com", "headers": {"key": "value1"}},
            {"url": "https://server2.com", "headers": {"key": "value2"}}
        ]
        self.mock_adapter = AsyncMock()
        self.discovery = A2AToolDiscovery(
            server_params=self.server_params,
            adapter=self.mock_adapter,
            connect_timeout=timedelta(seconds=30)
        )

    @pytest.mark.asyncio
    async def test_discover_and_adapt_tools_success(self):
        # Mock tools from both servers
        tools_server1 = [
            A2ATool("tool1", "Tool 1", {"type": "object"}),
            A2ATool("tool2", "Tool 2", {"type": "object"})
        ]
        tools_server2 = [
            A2ATool("tool3", "Tool 3", {"type": "object"})
        ]

        # Mock adapted tools
        adapted_tools = ["adapted_tool1", "adapted_tool2", "adapted_tool3"]
        self.mock_adapter.async_adapt.side_effect = adapted_tools

        with patch('src.a2a_core.a2a_session') as mock_session_context:
            # Create proper async context manager mocks
            def session_side_effect(params, timeout):
                if params["url"] == "https://server1.com":
                    mock_cm = AsyncMock()
                    mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), tools_server1))
                    mock_cm.__aexit__ = AsyncMock(return_value=None)
                    return mock_cm
                else:
                    mock_cm = AsyncMock()
                    mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), tools_server2))
                    mock_cm.__aexit__ = AsyncMock(return_value=None)
                    return mock_cm

            mock_session_context.side_effect = session_side_effect

            result = await self.discovery.discover_and_adapt_tools()

            assert len(result) == 3
            assert self.mock_adapter.async_adapt.call_count == 3

    @pytest.mark.asyncio
    async def test_discover_and_adapt_tools_partial_failure(self):
        # Mock one server failure, one success
        tools_server2 = [A2ATool("tool1", "Tool 1", {"type": "object"})]
        adapted_tool = "adapted_tool1"
        self.mock_adapter.async_adapt.return_value = adapted_tool

        with patch('src.a2a_core.a2a_session') as mock_session_context:
            def session_side_effect(params, timeout):
                if params["url"] == "https://server1.com":
                    raise Exception("Server 1 unavailable")
                else:
                    mock_cm = AsyncMock()
                    mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), tools_server2))
                    mock_cm.__aexit__ = AsyncMock(return_value=None)
                    return mock_cm

            mock_session_context.side_effect = session_side_effect

            result = await self.discovery.discover_and_adapt_tools()

            # Should get tools from server2 only
            assert len(result) == 1
            assert result[0] == adapted_tool

    @pytest.mark.asyncio
    async def test_discover_and_adapt_tools_all_failures(self):
        with patch('src.a2a_core.a2a_session') as mock_session_context:
            def session_side_effect(params, timeout):
                raise Exception("All servers down")

            mock_session_context.side_effect = session_side_effect

            result = await self.discovery.discover_and_adapt_tools()

            # Should return empty list if all servers fail
            assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__])