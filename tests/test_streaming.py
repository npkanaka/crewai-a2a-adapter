"""
Tests for Streaming functionality
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.streaming import (
    StreamingCallback,
    SimpleStreamingCallback,
    StreamingCrew,
)


class MockStreamingCallback(StreamingCallback):
    """Mock implementation of StreamingCallback for testing"""

    def __init__(self):
        self.events = []

    async def on_agent_start(self, agent_name: str, context: str):
        self.events.append(("agent_start", agent_name, context))

    async def on_agent_finish(self, agent_name: str, result: str):
        self.events.append(("agent_finish", agent_name, result))

    async def on_agent_thought(self, agent_name: str, thought: str):
        self.events.append(("agent_thought", agent_name, thought))

    async def on_tool_start(self, agent_name: str, tool_name: str, tool_input: str):
        self.events.append(("tool_start", agent_name, tool_name, tool_input))

    async def on_tool_finish(self, agent_name: str, tool_name: str, result: str):
        self.events.append(("tool_finish", agent_name, tool_name, result))

    async def on_crew_finish(self, final_result: str):
        self.events.append(("crew_finish", final_result))

    async def on_agent_delegation(self, from_agent: str, to_agent: str, task: str):
        self.events.append(("agent_delegation", from_agent, to_agent, task))

    async def on_coworker_interaction(
        self, agent_name: str, coworker_name: str, interaction_type: str, content: str
    ):
        self.events.append(("coworker_interaction", agent_name, coworker_name, interaction_type, content))

    async def on_max_iterations_exceeded(self, agent_name: str, iterations: int):
        self.events.append(("max_iterations_exceeded", agent_name, iterations))


class TestStreamingCallback:
    """Test the abstract StreamingCallback class"""

    @pytest.mark.asyncio
    async def test_mock_callback_implementation(self):
        callback = MockStreamingCallback()

        await callback.on_agent_start("test_agent", "test_context")
        await callback.on_agent_thought("test_agent", "thinking...")
        await callback.on_tool_start("test_agent", "test_tool", "input_data")
        await callback.on_tool_finish("test_agent", "test_tool", "output_data")
        await callback.on_agent_finish("test_agent", "agent_result")
        await callback.on_crew_finish("final_result")

        assert len(callback.events) == 6
        assert callback.events[0] == ("agent_start", "test_agent", "test_context")
        assert callback.events[1] == ("agent_thought", "test_agent", "thinking...")
        assert callback.events[2] == ("tool_start", "test_agent", "test_tool", "input_data")
        assert callback.events[3] == ("tool_finish", "test_agent", "test_tool", "output_data")
        assert callback.events[4] == ("agent_finish", "test_agent", "agent_result")
        assert callback.events[5] == ("crew_finish", "final_result")

    @pytest.mark.asyncio
    async def test_optional_callback_methods(self):
        callback = MockStreamingCallback()

        await callback.on_agent_delegation("manager", "worker", "task_description")
        await callback.on_coworker_interaction("agent1", "agent2", "question", "content")
        await callback.on_max_iterations_exceeded("agent", 100)

        assert len(callback.events) == 3
        assert callback.events[0] == ("agent_delegation", "manager", "worker", "task_description")
        assert callback.events[1] == ("coworker_interaction", "agent1", "agent2", "question", "content")
        assert callback.events[2] == ("max_iterations_exceeded", "agent", 100)


class TestSimpleStreamingCallback:
    """Test the SimpleStreamingCallback implementation"""

    def setup_method(self):
        self.callback = SimpleStreamingCallback(prefix="[TEST]")

    @pytest.mark.asyncio
    async def test_simple_callback_methods(self):
        # Test that methods don't raise exceptions
        await self.callback.on_agent_start("test_agent", "context")
        await self.callback.on_agent_finish("test_agent", "result")
        await self.callback.on_agent_thought("test_agent", "thought")
        await self.callback.on_tool_start("test_agent", "tool", "input")
        await self.callback.on_tool_finish("test_agent", "tool", "output")
        await self.callback.on_crew_finish("final")

    def test_default_prefix(self):
        default_callback = SimpleStreamingCallback()
        assert default_callback.prefix == "[CREW]"

    def test_custom_prefix(self):
        assert self.callback.prefix == "[TEST]"


class TestStreamingCrew:
    """Test the StreamingCrew wrapper"""

    def setup_method(self):
        # Create mock crew with agents and tasks
        self.mock_agent = MagicMock()
        self.mock_agent.role = "test_agent"
        self.mock_agent.tools = []

        self.mock_task = MagicMock()
        self.mock_task.description = "Test task description"
        self.mock_task.agent = self.mock_agent

        self.mock_crew = MagicMock()
        self.mock_crew.agents = [self.mock_agent]
        self.mock_crew.tasks = [self.mock_task]
        self.mock_crew.manager_agent = None

        self.callback = MockStreamingCallback()
        self.streaming_crew = StreamingCrew(self.mock_crew, self.callback)

    def test_streaming_crew_initialization(self):
        assert self.streaming_crew.crew == self.mock_crew
        assert self.streaming_crew.callback == self.callback
        assert self.streaming_crew._loop is None

    def test_format_tool_input(self):
        # Test with both args and kwargs
        result = self.streaming_crew._format_tool_input(
            ("arg1", "arg2"), {"param1": "value1", "param2": "value2"}
        )
        assert "arg1" in result
        assert "arg2" in result
        assert "param1=value1" in result
        assert "param2=value2" in result

        # Test with only args
        result = self.streaming_crew._format_tool_input(("single_arg",), {})
        assert "single_arg" in result

        # Test with only kwargs
        result = self.streaming_crew._format_tool_input((), {"key": "value"})
        assert "key=value" in result

        # Test with empty inputs
        result = self.streaming_crew._format_tool_input((), {})
        assert result == "()"

    def test_wrap_agent_tools(self):
        # Create mock tool with _run method
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        original_run = MagicMock(return_value="tool_result")
        mock_tool._run = original_run

        # Add tool to agent
        self.mock_agent.tools = [mock_tool]

        # Set up event loop for wrapper
        self.streaming_crew._loop = asyncio.new_event_loop()

        # Wrap the tools
        self.streaming_crew._wrap_agent_tools(self.mock_agent)

        # Verify the tool's _run method was replaced
        assert mock_tool._run != original_run

        # Test wrapped method execution
        with patch('asyncio.run_coroutine_threadsafe') as mock_run_coroutine:
            mock_tool._run("test_input")

            # Should have called original method
            original_run.assert_called_once_with("test_input")

            # Should have scheduled callbacks
            assert mock_run_coroutine.call_count >= 2  # At least start and finish callbacks

    def test_wrap_agent_tools_no_tools(self):
        # Test with agent that has no tools
        agent_no_tools = MagicMock()
        agent_no_tools.tools = None
        agent_no_tools.role = "empty_agent"

        # Should not raise exception
        self.streaming_crew._wrap_agent_tools(agent_no_tools)

        # Test with empty tools list
        agent_empty_tools = MagicMock()
        agent_empty_tools.tools = []
        agent_empty_tools.role = "empty_agent"

        self.streaming_crew._wrap_agent_tools(agent_empty_tools)

    @pytest.mark.asyncio
    async def test_kickoff_async_sequential_process(self):
        # Setup mock crew execution
        mock_result = MagicMock()
        mock_result.raw = "Final crew result"

        # Make kickoff_async return a coroutine
        async def mock_kickoff():
            return mock_result

        self.mock_crew.kickoff_async = mock_kickoff

        result = await self.streaming_crew.kickoff_async()

        # Verify result
        assert result == mock_result

        # Check that events were captured
        events = [event[0] for event in self.callback.events]
        assert "agent_start" in events
        assert "crew_finish" in events

    @pytest.mark.asyncio
    async def test_kickoff_async_hierarchical_process(self):
        # Setup hierarchical crew with manager
        mock_manager = MagicMock()
        mock_manager.role = "manager_agent"
        mock_manager.tools = []

        self.mock_crew.manager_agent = mock_manager

        mock_result = MagicMock()
        mock_result.raw = "Hierarchical result"

        async def mock_kickoff():
            return mock_result

        self.mock_crew.kickoff_async = mock_kickoff

        result = await self.streaming_crew.kickoff_async()

        assert result == mock_result

        # Check for delegation events
        events = [event[0] for event in self.callback.events]
        assert "agent_start" in events
        assert "agent_delegation" in events
        assert "agent_finish" in events
        assert "crew_finish" in events

    @pytest.mark.asyncio
    async def test_kickoff_async_with_exception(self):
        # Setup crew to raise exception
        async def mock_kickoff():
            raise Exception("Crew execution failed")

        self.mock_crew.kickoff_async = mock_kickoff

        with pytest.raises(Exception, match="Crew execution failed"):
            await self.streaming_crew.kickoff_async()

        # Should still have crew_finish event with error
        crew_finish_events = [event for event in self.callback.events if event[0] == "crew_finish"]
        assert len(crew_finish_events) == 1
        assert "Error:" in crew_finish_events[0][1]

    def test_synchronous_kickoff(self):
        # Setup mock crew
        mock_result = MagicMock()
        mock_result.raw = "Sync result"

        async def mock_kickoff():
            return mock_result

        self.mock_crew.kickoff_async = mock_kickoff

        with patch('asyncio.run') as mock_run:
            mock_run.return_value = mock_result
            result = self.streaming_crew.kickoff()

            assert result == mock_result
            mock_run.assert_called_once()

    def test_setup_delegation_tools(self):
        # Test that _wrap_delegation_tools doesn't crash
        # This method does complex patching, so we mainly test it doesn't fail
        try:
            self.streaming_crew._wrap_delegation_tools()
        except Exception as e:
            pytest.fail(f"_wrap_delegation_tools raised an exception: {e}")

    def test_wrap_delegation_tool(self):
        # Create mock delegation tool
        mock_tool = MagicMock()
        mock_tool.name = "Delegate work to coworker"
        original_run = MagicMock(return_value="delegation_result")
        mock_tool._run = original_run

        self.streaming_crew._loop = asyncio.new_event_loop()

        # Wrap the delegation tool
        self.streaming_crew._wrap_delegation_tool(mock_tool, "manager")

        # Verify the tool's _run method was replaced
        assert mock_tool._run != original_run

        # Test wrapped method execution
        with patch('asyncio.run_coroutine_threadsafe'):
            mock_tool._run("task", None, "worker")
            original_run.assert_called_once_with("task", None, "worker")


if __name__ == "__main__":
    pytest.main([__file__])