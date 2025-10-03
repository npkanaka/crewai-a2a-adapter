"""
Streaming Support for CrewAI

This module provides streaming capabilities for CrewAI workflows, allowing
real-time updates and progress tracking during agent execution.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StreamingCallback(ABC):
    """
    Abstract base class for streaming callbacks.

    Implement this class to receive real-time updates from CrewAI execution.
    """

    @abstractmethod
    async def on_agent_start(self, agent_name: str, context: str):
        """Called when an agent starts execution."""
        pass

    @abstractmethod
    async def on_agent_finish(self, agent_name: str, result: str):
        """Called when an agent finishes execution."""
        pass

    @abstractmethod
    async def on_agent_thought(self, agent_name: str, thought: str):
        """Called when an agent produces a thought."""
        pass

    @abstractmethod
    async def on_tool_start(self, agent_name: str, tool_name: str, tool_input: str):
        """Called when a tool execution starts."""
        pass

    @abstractmethod
    async def on_tool_finish(self, agent_name: str, tool_name: str, result: str):
        """Called when a tool execution finishes."""
        pass

    @abstractmethod
    async def on_crew_finish(self, final_result: str):
        """Called when the entire crew finishes execution."""
        pass

    async def on_agent_delegation(self, from_agent: str, to_agent: str, task: str):
        """Called when an agent delegates work to another agent."""
        pass

    async def on_coworker_interaction(
        self, agent_name: str, coworker_name: str, interaction_type: str, content: str
    ):
        """Called when agents interact with each other."""
        pass

    async def on_max_iterations_exceeded(self, agent_name: str, iterations: int):
        """Called when an agent exceeds maximum iterations."""
        pass


class SimpleStreamingCallback(StreamingCallback):
    """
    A simple implementation of StreamingCallback that logs all events.
    """

    def __init__(self, prefix: str = "[CREW]"):
        self.prefix = prefix

    async def on_agent_start(self, agent_name: str, context: str):
        logger.info("%s Agent '%s' started: %s", self.prefix, agent_name, context)

    async def on_agent_finish(self, agent_name: str, result: str):
        logger.info("%s Agent '%s' finished: %s", self.prefix, agent_name, result[:100])

    async def on_agent_thought(self, agent_name: str, thought: str):
        logger.info("%s Agent '%s' thinking: %s", self.prefix, agent_name, thought)

    async def on_tool_start(self, agent_name: str, tool_name: str, tool_input: str):
        logger.info(
            "%s Agent '%s' using tool '%s' with input: %s",
            self.prefix,
            agent_name,
            tool_name,
            tool_input[:100],
        )

    async def on_tool_finish(self, agent_name: str, tool_name: str, result: str):
        logger.info(
            "%s Agent '%s' tool '%s' result: %s", self.prefix, agent_name, tool_name, result[:100]
        )

    async def on_crew_finish(self, final_result: str):
        logger.info("%s Crew execution completed: %s", self.prefix, final_result[:200])


class StreamingCrew:
    """
    A wrapper around CrewAI's Crew class that adds streaming capabilities.

    This class intercepts agent and tool execution to provide real-time updates
    through callbacks.
    """

    def __init__(self, crew, callback: StreamingCallback):
        """
        Initialize the streaming crew wrapper.

        Args:
            crew: The CrewAI Crew instance to wrap
            callback: The callback to receive streaming updates
        """
        self.crew = crew
        self.callback = callback
        self._loop = None

        # Set up monitoring
        self._setup_thought_capture()
        self._setup_tool_wrapping()

    def _setup_thought_capture(self):
        """Set up step callback to capture agent thoughts."""

        def step_callback(formatted_answer):
            """Capture thoughts from agent execution."""
            try:
                # Check if this is an AgentAction with a thought
                if hasattr(formatted_answer, "thought") and hasattr(formatted_answer, "tool"):
                    thought = formatted_answer.thought
                    if thought and thought.strip():
                        agent_name = "Agent"  # Default fallback

                        # Try to get current agent name
                        if hasattr(self.crew, "agents") and self.crew.agents:
                            agent_name = self.crew.agents[0].role

                        # Queue the thought message
                        if self._loop:
                            asyncio.run_coroutine_threadsafe(
                                self.callback.on_agent_thought(agent_name, thought.strip()),
                                self._loop,
                            )
            except Exception as e:
                logger.error(f"Error capturing thought: {e}")

        self._thought_callback = step_callback

    def _setup_tool_wrapping(self):
        """Set up tool execution monitoring."""
        # This will be called before execution to wrap all tools
        pass

    def _wrap_agent_tools(self, agent):
        """
        Wrap an agent's tools to intercept execution.

        Args:
            agent: The agent whose tools to wrap
        """
        if not hasattr(agent, "tools") or not agent.tools:
            return

        agent_role = getattr(agent, "role", "Agent")

        for tool in agent.tools:
            if hasattr(tool, "_run"):
                original_run = tool._run

                def create_wrapper(original_method, tool_instance, agent_name):
                    def wrapper(*args, **kwargs):
                        # Extract input information
                        tool_input = self._format_tool_input(args, kwargs)

                        # Report tool start
                        if self._loop:
                            asyncio.run_coroutine_threadsafe(
                                self.callback.on_tool_start(
                                    agent_name, tool_instance.name, tool_input
                                ),
                                self._loop,
                            )

                        try:
                            # Execute the original tool
                            result = original_method(*args, **kwargs)

                            # Report tool finish
                            if self._loop:
                                asyncio.run_coroutine_threadsafe(
                                    self.callback.on_tool_finish(
                                        agent_name, tool_instance.name, str(result)
                                    ),
                                    self._loop,
                                )

                            return result

                        except Exception as e:
                            # Report tool error
                            if self._loop:
                                error_msg = f"Tool error: {str(e)}"
                                asyncio.run_coroutine_threadsafe(
                                    self.callback.on_tool_finish(
                                        agent_name, tool_instance.name, error_msg
                                    ),
                                    self._loop,
                                )
                            raise

                    return wrapper

                # Replace the tool's _run method with our wrapper
                tool._run = create_wrapper(original_run, tool, agent_role)

    def _format_tool_input(self, args: tuple, kwargs: dict) -> str:
        """Format tool input for display."""
        parts = []
        if args:
            parts.extend(str(arg)[:50] for arg in args)
        if kwargs:
            parts.extend(f"{k}={str(v)[:50]}" for k, v in kwargs.items())
        return f"({', '.join(parts)})"

    def _wrap_delegation_tools(self):
        """Set up monitoring for delegation tools."""
        try:
            # Patch the crew's delegation tool injection
            if hasattr(self.crew, "_inject_delegation_tools"):
                original_inject = self.crew._inject_delegation_tools

                def patched_inject(tools, task_agent, agents):
                    # Call original method
                    updated_tools = original_inject(tools, task_agent, agents)

                    # Wrap delegation tools
                    for tool in updated_tools:
                        if tool.name in ["Ask question to coworker", "Delegate work to coworker"]:
                            self._wrap_delegation_tool(tool, task_agent.role)

                    return updated_tools

                self.crew._inject_delegation_tools = patched_inject

        except Exception as e:
            logger.error(f"Failed to patch delegation tools: {e}")

    def _wrap_delegation_tool(self, tool, agent_role: str):
        """Wrap a delegation tool to capture coworker interactions."""
        if hasattr(tool, "_run"):
            original_run = tool._run

            def wrapper(*args, **kwargs):
                # Extract interaction details
                coworker_name = "Unknown"
                content = "Unknown"

                if len(args) >= 3:
                    content = str(args[0])
                    coworker_name = str(args[2]) if args[2] else "Unknown"
                elif kwargs:
                    content = str(kwargs.get("task", kwargs.get("question", "Unknown")))
                    coworker_name = str(kwargs.get("coworker", "Unknown"))

                # Report interaction
                if self._loop:
                    asyncio.run_coroutine_threadsafe(
                        self.callback.on_coworker_interaction(
                            agent_role, coworker_name, tool.name, content
                        ),
                        self._loop,
                    )

                # Execute original tool
                return original_run(*args, **kwargs)

            tool._run = wrapper

    async def kickoff_async(self):
        """
        Execute the CrewAI workflow with streaming updates.

        Returns:
            The crew execution result
        """
        # Store event loop for callbacks
        self._loop = asyncio.get_event_loop()

        # Wrap all agent tools
        for agent in self.crew.agents:
            self._wrap_agent_tools(agent)

            # Set step callback for thoughts
            if hasattr(agent, "step_callback"):
                agent.step_callback = self._thought_callback

        # Set up delegation tool monitoring
        self._wrap_delegation_tools()

        try:
            # Handle hierarchical process
            if hasattr(self.crew, "manager_agent") and self.crew.manager_agent:
                self._wrap_agent_tools(self.crew.manager_agent)

                if hasattr(self.crew.manager_agent, "step_callback"):
                    self.crew.manager_agent.step_callback = self._thought_callback

                # Report manager start
                await self.callback.on_agent_start(
                    self.crew.manager_agent.role, f"Coordinating {len(self.crew.tasks)} tasks"
                )

                # Report delegations
                for i, task in enumerate(self.crew.tasks, 1):
                    if task.agent:
                        await self.callback.on_agent_delegation(
                            self.crew.manager_agent.role,
                            task.agent.role,
                            f"Task {i}: {task.description[:100]}",
                        )

                # Execute crew
                result = await self.crew.kickoff_async()

                # Report manager finish
                await self.callback.on_agent_finish(
                    self.crew.manager_agent.role, "Coordination completed"
                )

            else:
                # Sequential process
                for i, task in enumerate(self.crew.tasks, 1):
                    agent = task.agent if task.agent else self.crew.agents[0]

                    await self.callback.on_agent_start(
                        agent.role, f"Task {i}/{len(self.crew.tasks)}: {task.description[:100]}"
                    )

                # Execute crew
                result = await self.crew.kickoff_async()

                # Report agent completions
                for i, task in enumerate(self.crew.tasks, 1):
                    agent = task.agent if task.agent else self.crew.agents[0]
                    await self.callback.on_agent_finish(
                        agent.role, f"Completed task {i}/{len(self.crew.tasks)}"
                    )

            # Report crew finish
            final_output = getattr(result, "raw", str(result))
            await self.callback.on_crew_finish(final_output)

            return result

        except Exception as e:
            logger.error(f"Error in streaming crew execution: {e}")
            await self.callback.on_crew_finish(f"Error: {str(e)}")
            raise

    def kickoff(self):
        """
        Synchronous execution method.

        Returns:
            The crew execution result
        """
        return asyncio.run(self.kickoff_async())
