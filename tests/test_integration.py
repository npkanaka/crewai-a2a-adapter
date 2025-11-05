"""
Integration tests for the CrewAI A2A Adapter
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import (
    A2ATool,
    A2ASession,
    A2ACrewAIAdapter,
    CrewAIToolkit,
    StreamingCallback,
    StreamingCrew,
)


class TestIntegrationCallback(StreamingCallback):
    """Test callback that captures all events"""

    def __init__(self):
        self.events = []

    async def on_agent_start(self, agent_name: str, context: str):
        self.events.append(f"AGENT_START: {agent_name}")

    async def on_agent_finish(self, agent_name: str, result: str):
        self.events.append(f"AGENT_FINISH: {agent_name}")

    async def on_agent_thought(self, agent_name: str, thought: str):
        self.events.append(f"THOUGHT: {agent_name} - {thought[:50]}")

    async def on_tool_start(self, agent_name: str, tool_name: str, tool_input: str):
        self.events.append(f"TOOL_START: {tool_name}")

    async def on_tool_finish(self, agent_name: str, tool_name: str, result: str):
        self.events.append(f"TOOL_FINISH: {tool_name} - {str(result)[:50]}")

    async def on_crew_finish(self, final_result: str):
        self.events.append(f"CREW_FINISH: {str(final_result)[:50]}")

    async def on_agent_delegation(self, from_agent: str, to_agent: str, task: str):
        self.events.append(f"DELEGATION: {from_agent} -> {to_agent}")

    async def on_coworker_interaction(
        self, agent_name: str, coworker_name: str, interaction_type: str, content: str
    ):
        self.events.append(f"INTERACTION: {agent_name} -> {coworker_name}")

    async def on_max_iterations_exceeded(self, agent_name: str, iterations: int):
        self.events.append(f"MAX_ITERATIONS: {agent_name} ({iterations})")


class TestEndToEndIntegration:
    """Test complete end-to-end workflows"""

    @pytest.mark.asyncio
    @patch('src.a2a_core.httpx.AsyncClient')
    @patch('src.a2a_core.A2ACardResolver')
    @patch('src.a2a_core.A2AClient')
    async def test_complete_toolkit_workflow(self, mock_a2a_client, mock_resolver_class, mock_client_class):
        """Test the complete workflow from A2A discovery to CrewAI execution"""

        # Mock A2A server response
        mock_skill = MagicMock()
        mock_skill.id = "research_skill"
        mock_skill.description = "A research skill"

        mock_agent_card = MagicMock()
        mock_agent_card.skills = [mock_skill]

        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        mock_resolver = AsyncMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.get_agent_card.return_value = mock_agent_card

        mock_a2a = AsyncMock()
        mock_a2a_client.return_value = mock_a2a

        # Create toolkit and load tools
        toolkit = CrewAIToolkit()
        server_configs = [
            {
                "url": "https://test-agent.com",
                "headers": {"Authorization": "Bearer test-token"}
            }
        ]

        await toolkit.load_from_a2a_servers(server_configs)

        # Verify tool was loaded
        assert len(toolkit.get_tools()) == 1
        tool = toolkit.get_tools()[0]
        assert tool.name == "research_skill"
        assert "research skill" in tool.description.lower()

        # Verify tool names list
        tool_names = toolkit.list_tools()
        assert "research_skill" in tool_names

    @pytest.mark.asyncio
    async def test_toolkit_multiple_servers(self):
        """Test toolkit loading from multiple A2A servers"""

        with patch('src.a2a_core.a2a_session') as mock_session_context:
            # Mock responses from different servers
            server1_tools = [
                A2ATool("tool1", "Tool from server 1", {"type": "object"}),
                A2ATool("tool2", "Another tool from server 1", {"type": "object"})
            ]
            server2_tools = [
                A2ATool("tool3", "Tool from server 2", {"type": "object"})
            ]

            async def session_side_effect(params, timeout):
                mock_session = AsyncMock()
                if "server1" in params["url"]:
                    tools = server1_tools
                else:
                    tools = server2_tools

                return AsyncMock(
                    __aenter__=AsyncMock(return_value=(mock_session, tools)),
                    __aexit__=AsyncMock(return_value=None)
                )

            mock_session_context.side_effect = session_side_effect

            # Create toolkit and load from multiple servers
            toolkit = CrewAIToolkit()
            server_configs = [
                {"url": "https://server1.com"},
                {"url": "https://server2.com"}
            ]

            await toolkit.load_from_a2a_servers(server_configs)

            # Should have tools from both servers
            assert len(toolkit.get_tools()) == 3
            tool_names = toolkit.list_tools()
            assert "tool1" in tool_names
            assert "tool2" in tool_names
            assert "tool3" in tool_names

    @pytest.mark.asyncio
    async def test_toolkit_server_failure_resilience(self):
        """Test that toolkit handles individual server failures gracefully"""

        with patch('src.a2a_core.a2a_session') as mock_session_context:
            async def session_side_effect(params, timeout):
                if "failing-server" in params["url"]:
                    raise Exception("Server unavailable")
                else:
                    # Working server
                    tools = [A2ATool("working_tool", "Tool from working server", {"type": "object"})]
                    mock_session = AsyncMock()
                    return AsyncMock(
                        __aenter__=AsyncMock(return_value=(mock_session, tools)),
                        __aexit__=AsyncMock(return_value=None)
                    )

            mock_session_context.side_effect = session_side_effect

            toolkit = CrewAIToolkit()
            server_configs = [
                {"url": "https://failing-server.com"},
                {"url": "https://working-server.com"}
            ]

            await toolkit.load_from_a2a_servers(server_configs)

            # Should have tool from working server only
            assert len(toolkit.get_tools()) == 1
            assert toolkit.get_tools()[0].name == "working_tool"

    def test_streaming_crew_integration(self):
        """Test StreamingCrew integration with mock CrewAI components"""

        # Create mock CrewAI components
        mock_agent = MagicMock()
        mock_agent.role = "Research Agent"
        mock_agent.tools = []

        mock_task = MagicMock()
        mock_task.description = "Research the latest AI developments"
        mock_task.agent = mock_agent

        mock_crew = MagicMock()
        mock_crew.agents = [mock_agent]
        mock_crew.tasks = [mock_task]
        mock_crew.manager_agent = None

        callback = TestIntegrationCallback()
        streaming_crew = StreamingCrew(mock_crew, callback)

        # Test initialization
        assert streaming_crew.crew == mock_crew
        assert streaming_crew.callback == callback

        # Test tool wrapping doesn't crash
        streaming_crew._wrap_agent_tools(mock_agent)

    @pytest.mark.asyncio
    async def test_adapter_with_complex_schema(self):
        """Test adapter with complex JSON schema"""

        complex_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "minimum": 1,
                    "maximum": 100
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"}
                            }
                        }
                    }
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include metadata"
                }
            },
            "required": ["query"]
        }

        a2a_tool = A2ATool(
            name="complex_search",
            description="A complex search tool",
            input_schema=complex_schema
        )

        adapter = A2ACrewAIAdapter()
        server_params = {"url": "https://test.com"}

        crewai_tool = await adapter.async_adapt(a2a_tool, server_params)

        # Test that the tool was created successfully
        assert crewai_tool.name == "complex_search"
        assert hasattr(crewai_tool, 'args_schema')

        # Test that the schema conversion worked
        schema_dict = crewai_tool.args_schema.model_json_schema()
        assert "query" in schema_dict["properties"]
        assert "limit" in schema_dict["properties"]
        assert "filters" in schema_dict["properties"]
        assert "include_metadata" in schema_dict["properties"]

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling throughout the workflow"""

        # Test toolkit creation with invalid server
        toolkit = CrewAIToolkit()

        with patch('src.a2a_core.a2a_session') as mock_session:
            mock_session.side_effect = Exception("Network error")

            # Should not raise exception, just log and continue
            await toolkit.load_from_a2a_servers([{"url": "https://invalid.com"}])

            # Should have no tools loaded
            assert len(toolkit.get_tools()) == 0

    @pytest.mark.asyncio
    @patch('src.crewai_adapter.A2ASession')
    async def test_tool_execution_edge_cases(self, mock_session_class):
        """Test edge cases in tool execution"""

        # Test tool with None result
        mock_session = AsyncMock()
        from src.a2a_core import A2ACallToolResult
        mock_session.call_tool.return_value = A2ACallToolResult(
            success=True,
            result=None
        )
        mock_session_class.return_value = mock_session

        adapter = A2ACrewAIAdapter()
        a2a_tool = A2ATool(
            name="null_result_tool",
            description="Tool that returns None",
            input_schema={"type": "object"}
        )

        crewai_tool = await adapter.async_adapt(a2a_tool, {"url": "https://test.com"})
        result = await crewai_tool._arun()

        # Should handle None result gracefully
        assert result == "Success"

    def test_callback_event_ordering(self):
        """Test that streaming callbacks maintain proper event ordering"""

        callback = TestIntegrationCallback()

        # Simulate a sequence of events
        async def simulate_events():
            await callback.on_agent_start("Agent1", "Starting work")
            await callback.on_tool_start("Agent1", "Tool1", "input")
            await callback.on_tool_finish("Agent1", "Tool1", "output")
            await callback.on_agent_finish("Agent1", "completed")
            await callback.on_crew_finish("All done")

        asyncio.run(simulate_events())

        # Verify event order
        expected_order = [
            "AGENT_START: Agent1",
            "TOOL_START: Tool1",
            "TOOL_FINISH: Tool1",
            "AGENT_FINISH: Agent1",
            "CREW_FINISH: All done"
        ]

        assert callback.events == expected_order

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test concurrent execution of multiple adapted tools"""

        with patch('src.crewai_adapter.A2ASession') as mock_session_class:
            from src.a2a_core import A2ACallToolResult

            # Mock session that simulates different execution times
            async def mock_call_tool(tool_name, args):
                if tool_name == "fast_tool":
                    await asyncio.sleep(0.01)
                    return A2ACallToolResult(True, "Fast result")
                else:
                    await asyncio.sleep(0.02)
                    return A2ACallToolResult(True, "Slow result")

            mock_session = AsyncMock()
            mock_session.call_tool.side_effect = mock_call_tool
            mock_session_class.return_value = mock_session

            # Create multiple tools
            adapter = A2ACrewAIAdapter()
            tools = []

            for tool_name in ["fast_tool", "slow_tool"]:
                a2a_tool = A2ATool(
                    name=tool_name,
                    description=f"A {tool_name}",
                    input_schema={"type": "object"}
                )
                crewai_tool = await adapter.async_adapt(a2a_tool, {"url": "https://test.com"})
                tools.append(crewai_tool)

            # Execute tools concurrently
            tasks = [tool._arun() for tool in tools]
            results = await asyncio.gather(*tasks)

            assert "Fast result" in results
            assert "Slow result" in results

    def test_memory_cleanup(self):
        """Test that resources are properly cleaned up"""

        # Create and destroy multiple toolkits
        for i in range(10):
            toolkit = CrewAIToolkit()

            # Add some mock tools
            for j in range(5):
                mock_tool = MagicMock()
                mock_tool.name = f"tool_{i}_{j}"
                toolkit.add_tool(mock_tool)

            # Clear the toolkit
            toolkit.tools.clear()

        # Test should complete without memory issues
        assert True


if __name__ == "__main__":
    pytest.main([__file__])