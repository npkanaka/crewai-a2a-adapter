"""
Tests for CrewAI Adapter functionality
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.a2a_core import A2ATool, A2ACallToolResult
from src.crewai_adapter import (
    A2ACrewAIAdapter,
    CrewAIToolkit,
    create_pydantic_model_from_schema,
)


class TestPydanticModelCreation:
    """Test Pydantic model creation from JSON schemas"""

    def test_create_model_with_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name field"},
                "age": {"type": "integer", "description": "The age field"},
                "active": {"type": "boolean", "description": "Active status"}
            },
            "required": ["name", "age"]
        }

        model_class = create_pydantic_model_from_schema(schema, "TestModel")

        # Test model creation
        instance = model_class(name="John", age=30, active=True)
        assert instance.name == "John"
        assert instance.age == 30
        assert instance.active is True

        # Test required fields
        with pytest.raises(Exception):  # Should fail without required fields
            model_class()

        # Test optional fields
        instance2 = model_class(name="Jane", age=25)
        assert instance2.active is None

    def test_create_model_empty_schema(self):
        model_class = create_pydantic_model_from_schema({}, "EmptyModel")
        instance = model_class()
        assert instance is not None

    def test_create_model_no_properties(self):
        schema = {"type": "object"}
        model_class = create_pydantic_model_from_schema(schema, "NoPropsModel")
        instance = model_class()
        assert instance is not None

    def test_create_model_various_types(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "number": {"type": "number"},
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
                "data": {"type": "object"},
                "items": {"type": "array"}
            }
        }

        model_class = create_pydantic_model_from_schema(schema, "VariousTypesModel")
        instance = model_class(
            text="hello",
            number=3.14,
            count=42,
            flag=True,
            data={"key": "value"},
            items=[1, 2, 3]
        )

        assert instance.text == "hello"
        assert instance.number == 3.14
        assert instance.count == 42
        assert instance.flag is True
        assert instance.data == {"key": "value"}
        assert instance.items == [1, 2, 3]


class TestA2ACrewAIAdapter:
    """Test the A2A to CrewAI adapter"""

    def setup_method(self):
        self.adapter = A2ACrewAIAdapter()
        self.server_params = {
            "url": "https://test-server.com",
            "headers": {"Authorization": "Bearer test-token"}
        }

    @pytest.mark.asyncio
    async def test_adapt_simple_tool(self):
        a2a_tool = A2ATool(
            name="simple_tool",
            description="A simple test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input text"}
                },
                "required": ["input"]
            }
        )

        crewai_tool = await self.adapter.async_adapt(a2a_tool, self.server_params)

        # Test basic properties
        assert crewai_tool.name == "simple_tool"
        assert "A simple test tool" in crewai_tool.description
        assert hasattr(crewai_tool, '_run')
        assert hasattr(crewai_tool, '_arun')

        # Test args schema
        assert issubclass(crewai_tool.args_schema, BaseModel)

    @pytest.mark.asyncio
    async def test_adapt_tool_with_no_schema(self):
        a2a_tool = A2ATool(
            name="no_schema_tool",
            description="Tool without schema",
            input_schema={}
        )

        crewai_tool = await self.adapter.async_adapt(a2a_tool, self.server_params)
        assert crewai_tool.name == "no_schema_tool"

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2ASession')
    async def test_tool_execution_success(self, mock_session_class):
        # Setup mock session
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = A2ACallToolResult(
            success=True,
            result="Tool executed successfully"
        )
        mock_session_class.return_value = mock_session

        # Create and adapt tool
        a2a_tool = A2ATool(
            name="test_tool",
            description="Test tool",
            input_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            }
        )

        crewai_tool = await self.adapter.async_adapt(a2a_tool, self.server_params)

        # Test async execution
        result = await crewai_tool._arun(input="test input")
        assert result == "Tool executed successfully"

        # Verify session interactions
        mock_session.connect.assert_called_once()
        mock_session.call_tool.assert_called_once_with("test_tool", {"input": "test input"})
        mock_session.disconnect.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2ASession')
    async def test_tool_execution_failure(self, mock_session_class):
        # Setup mock session with failure
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = A2ACallToolResult(
            success=False,
            result=None,
            error="Tool execution failed"
        )
        mock_session_class.return_value = mock_session

        # Create and adapt tool
        a2a_tool = A2ATool(
            name="failing_tool",
            description="A tool that fails",
            input_schema={"type": "object"}
        )

        crewai_tool = await self.adapter.async_adapt(a2a_tool, self.server_params)

        # Test execution
        result = await crewai_tool._arun()
        assert "failed" in result.lower()
        assert "Tool execution failed" in result

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2ASession')
    async def test_tool_execution_exception(self, mock_session_class):
        # Setup mock session that raises exception
        mock_session = AsyncMock()
        mock_session.connect.side_effect = Exception("Connection error")
        mock_session_class.return_value = mock_session

        # Create and adapt tool
        a2a_tool = A2ATool(
            name="error_tool",
            description="Tool that raises exception",
            input_schema={"type": "object"}
        )

        crewai_tool = await self.adapter.async_adapt(a2a_tool, self.server_params)

        # Test execution
        result = await crewai_tool._arun()
        assert "error" in result.lower()
        assert "Connection error" in result

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2ASession')
    def test_sync_execution_with_new_loop(self, mock_session_class):
        # Setup mock session
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = A2ACallToolResult(
            success=True,
            result="Sync execution successful"
        )
        mock_session_class.return_value = mock_session

        # Create and adapt tool
        a2a_tool = A2ATool(
            name="sync_tool",
            description="Tool for sync test",
            input_schema={"type": "object"}
        )

        async def create_and_test_tool():
            crewai_tool = await self.adapter.async_adapt(a2a_tool, self.server_params)
            # Test sync execution (should work when no event loop is running)
            with patch('asyncio.get_running_loop', side_effect=RuntimeError("No loop")):
                with patch('asyncio.run') as mock_run:
                    mock_run.return_value = "Mocked sync result"
                    result = crewai_tool._run()
                    assert result == "Mocked sync result"

        asyncio.run(create_and_test_tool())

    @pytest.mark.asyncio
    async def test_sync_execution_with_running_loop(self):
        a2a_tool = A2ATool(
            name="sync_tool_running_loop",
            description="Tool for running loop test",
            input_schema={"type": "object"}
        )

        with patch('src.crewai_adapter.A2ASession') as mock_session_class, \
             patch('src.crewai_adapter.asyncio.get_running_loop') as mock_get_loop, \
             patch('src.crewai_adapter.asyncio.run') as mock_asyncio_run, \
             patch('src.crewai_adapter.threading.Thread') as mock_thread_class:

            mock_session = AsyncMock()
            mock_session.call_tool.return_value = A2ACallToolResult(
                success=True,
                result="unused"
            )
            mock_session_class.return_value = mock_session

            crewai_tool = await self.adapter.async_adapt(a2a_tool, self.server_params)

            async_result = "threaded result"
            crewai_tool._execute_async = AsyncMock(return_value=async_result)

            mock_get_loop.return_value = MagicMock()
            mock_asyncio_run.return_value = async_result

            class ImmediateThread:
                def __init__(self, target, daemon=True):
                    self._target = target
                def start(self):
                    self._target()
                def join(self):
                    return None

            mock_thread_class.side_effect = lambda target, daemon=True: ImmediateThread(target, daemon)

            result = crewai_tool._run()

            assert result == async_result
            crewai_tool._execute_async.assert_called_once_with()
            mock_asyncio_run.assert_called_once()
            mock_thread_class.assert_called_once()


class TestCrewAIToolkit:
    """Test the CrewAI Toolkit class"""

    def setup_method(self):
        self.toolkit = CrewAIToolkit()

    def test_toolkit_initialization(self):
        assert len(self.toolkit.tools) == 0
        assert self.toolkit.get_tools() == []

    def test_toolkit_with_initial_tools(self):
        mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
        toolkit = CrewAIToolkit(tools=mock_tools)
        assert len(toolkit.tools) == 2

    def test_add_tool(self):
        mock_tool = MagicMock(name="test_tool")
        self.toolkit.add_tool(mock_tool)

        assert len(self.toolkit.tools) == 1
        assert self.toolkit.tools[0] == mock_tool

    def test_get_tool_by_name(self):
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        self.toolkit.add_tool(mock_tool1)
        self.toolkit.add_tool(mock_tool2)

        found_tool = self.toolkit.get_tool_by_name("tool1")
        assert found_tool == mock_tool1

        not_found = self.toolkit.get_tool_by_name("nonexistent")
        assert not_found is None

    def test_list_tools(self):
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        self.toolkit.add_tool(mock_tool1)
        self.toolkit.add_tool(mock_tool2)

        tool_names = self.toolkit.list_tools()
        assert tool_names == ["tool1", "tool2"]

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2AToolDiscovery')
    async def test_load_from_a2a_servers(self, mock_discovery_class):
        # Setup mock discovery
        mock_tool1 = MagicMock(name="discovered_tool1")
        mock_tool2 = MagicMock(name="discovered_tool2")
        mock_discovery = AsyncMock()
        mock_discovery.discover_and_adapt_tools.return_value = [mock_tool1, mock_tool2]
        mock_discovery_class.return_value = mock_discovery

        server_configs = [
            {"url": "https://server1.com"},
            {"url": "https://server2.com"}
        ]

        await self.toolkit.load_from_a2a_servers(server_configs)

        # Verify tools were added
        assert len(self.toolkit.tools) == 2
        assert mock_tool1 in self.toolkit.tools
        assert mock_tool2 in self.toolkit.tools

        # Verify discovery was configured correctly
        mock_discovery_class.assert_called_once()
        call_args = mock_discovery_class.call_args
        assert call_args[1]['server_params'] == server_configs
        assert isinstance(call_args[1]['adapter'], A2ACrewAIAdapter)

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2AToolDiscovery')
    async def test_load_from_a2a_servers_custom_adapter(self, mock_discovery_class):
        # Setup custom adapter
        custom_adapter = A2ACrewAIAdapter()
        mock_discovery = AsyncMock()
        mock_discovery.discover_and_adapt_tools.return_value = []
        mock_discovery_class.return_value = mock_discovery

        await self.toolkit.load_from_a2a_servers(
            server_configs=[{"url": "https://test.com"}],
            adapter=custom_adapter
        )

        # Verify custom adapter was used
        call_args = mock_discovery_class.call_args
        assert call_args[1]['adapter'] == custom_adapter

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2AToolDiscovery')
    async def test_load_from_a2a_servers_empty_result(self, mock_discovery_class):
        # Setup discovery with no tools
        mock_discovery = AsyncMock()
        mock_discovery.discover_and_adapt_tools.return_value = []
        mock_discovery_class.return_value = mock_discovery

        initial_count = len(self.toolkit.tools)
        await self.toolkit.load_from_a2a_servers([{"url": "https://empty.com"}])

        # Should not add any tools
        assert len(self.toolkit.tools) == initial_count


if __name__ == "__main__":
    pytest.main([__file__])