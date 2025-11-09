"""
CrewAI Adapter for A2A Tools

This module provides an adapter that converts A2A tools into CrewAI-compatible tools,
enabling seamless integration between A2A agents and CrewAI workflows.
"""

import asyncio
import logging
import threading
from typing import Any, Type

import jsonref
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from .a2a_core import (
    A2ATool,
    A2ASession,
    ToolAdapter,
    A2AToolDiscovery,
)

logger = logging.getLogger(__name__)


# JSON Schema to Python type mapping
JSON_TYPE_MAPPING = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def create_pydantic_model_from_schema(
    schema: dict, model_name: str = "A2AToolInput"
) -> Type[BaseModel]:
    """
    Create a Pydantic model from a JSON schema.

    Args:
        schema: JSON schema dictionary
        model_name: Name for the generated model class

    Returns:
        A dynamically created Pydantic model class
    """
    if not schema or "properties" not in schema:
        # Return an empty model if no schema is provided
        return create_model(model_name)

    properties = schema["properties"]
    required = set(schema.get("required", []))

    # Build field definitions for Pydantic
    field_definitions = {}

    for field_name, field_schema in properties.items():
        # Get the base type
        field_type = JSON_TYPE_MAPPING.get(field_schema.get("type", "string"), str)

        # Get field description
        description = field_schema.get("description", "")

        # Determine if field is required
        if field_name in required:
            field_definitions[field_name] = (field_type, Field(description=description))
        else:
            # Optional field
            field_definitions[field_name] = (
                field_type | None,
                Field(default=None, description=description),
            )

    # Create and return the model
    return create_model(model_name, **field_definitions)


class A2ACrewAIAdapter(ToolAdapter):
    """
    Adapter that converts A2A tools into CrewAI-compatible tools.

    This adapter handles:
    - Schema conversion from JSON Schema to Pydantic models
    - Async-to-sync bridging (CrewAI uses synchronous tools)
    - Error handling and logging
    - Tool description enhancement
    """

    async def async_adapt(
        self,
        a2a_tool: A2ATool,
        server_params: dict[str, Any],
    ) -> BaseTool:
        """
        Adapt an A2A tool to a CrewAI tool.

        Args:
            a2a_tool: The A2A tool to adapt
            server_params: Connection parameters for the A2A server

        Returns:
            A CrewAI-compatible tool instance
        """
        # Create Pydantic model from the tool's input schema
        tool_input_model = create_pydantic_model_from_schema(
            a2a_tool.input_schema, model_name=f"{a2a_tool.name.replace(' ', '')}Input"
        )

        class A2ACrewAITool(BaseTool):
            """Dynamically created CrewAI tool wrapping an A2A tool."""

            name: str = a2a_tool.name
            description: str = a2a_tool.description or f"A2A tool: {a2a_tool.name}"
            args_schema: Type[BaseModel] = tool_input_model
            _server_params: dict[str, Any] = server_params
            _tool_name: str = a2a_tool.name

            async def _execute_async(self, *args: Any, **kwargs: Any) -> Any:
                """Execute the A2A tool asynchronously."""
                logger.debug(
                    "Executing A2A tool '%s' with kwargs: %s",
                    self._tool_name,
                    list(kwargs.keys()),
                )

                session = A2ASession(
                    server_url=self._server_params["url"],
                    headers=self._server_params.get("headers"),
                )

                try:
                    await session.connect()
                    result = await session.call_tool(self._tool_name, kwargs)

                    if result.success:
                        logger.info("A2A tool '%s' executed successfully", self._tool_name)
                        return str(result.result) if result.result is not None else "Success"
                    else:
                        error_msg = f"A2A tool '{self._tool_name}' failed: {result.error}"
                        logger.error(error_msg)
                        return error_msg

                except Exception as e:
                    error_msg = f"Error executing A2A tool '{self._tool_name}': {str(e)}"
                    logger.exception(error_msg)
                    return error_msg

                finally:
                    await session.disconnect()

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                """
                Synchronous execution method required by CrewAI.

                This method bridges async A2A tools to sync CrewAI environment.
                """
                logger.debug(
                    "Bridging sync call to async for A2A tool '%s'",
                    self._tool_name,
                )

                try:
                    # Try to reuse an active event loop when possible
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        # No event loop running, execute directly
                        return asyncio.run(self._execute_async(*args, **kwargs))

                    # We are inside an active loop; run the async work in a helper thread
                    result_container: dict[str, Any] = {}
                    error_container: dict[str, BaseException] = {}

                    def runner() -> None:
                        try:
                            result_container["value"] = asyncio.run(
                                self._execute_async(*args, **kwargs)
                            )
                        except BaseException as runner_exc:  # noqa: BLE001
                            error_container["error"] = runner_exc

                    thread = threading.Thread(target=runner, daemon=True)
                    thread.start()
                    thread.join()

                    if error_container:
                        raise error_container["error"]

                    return result_container.get("value")

                except Exception as e:
                    error_msg = f"Error in sync wrapper for A2A tool '{self._tool_name}': {str(e)}"
                    logger.error(error_msg)
                    return error_msg

            async def _arun(self, *args: Any, **kwargs: Any) -> Any:
                """
                Native async execution method.

                Some versions of CrewAI support async tools through this method.
                """
                logger.debug(
                    "Native async execution of A2A tool '%s'",
                    self._tool_name,
                )
                return await self._execute_async(*args, **kwargs)

            def _generate_description(self):
                """
                Generate an enhanced description including schema information.

                This helps LLMs understand how to use the tool properly.
                """
                try:
                    # Resolve refs and clean up the schema for display
                    schema = jsonref.replace_refs(self.args_schema.model_json_schema())

                    # Remove internal fields
                    clean_schema = {k: v for k, v in schema.items() if k not in ["$defs", "title"]}

                    # Update description with schema info
                    self.description = (
                        f"Tool: {self.name}\n"
                        f"Description: {a2a_tool.description or 'No description provided'}\n"
                        f"Arguments: {clean_schema}"
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to generate enhanced description for tool '%s': %s",
                        self.name,
                        str(e),
                    )

        # Create an instance of the tool
        tool_instance = A2ACrewAITool()

        # Generate enhanced description
        tool_instance._generate_description()

        return tool_instance


class CrewAIToolkit:
    """
    A toolkit that manages multiple A2A tools for use in CrewAI.
    """

    def __init__(self, tools: list[BaseTool] = None):
        """
        Initialize the toolkit.

        Args:
            tools: Optional list of pre-adapted tools
        """
        self.tools = tools or []

    def add_tool(self, tool: BaseTool):
        """Add a tool to the toolkit."""
        self.tools.append(tool)
        logger.info("Added tool '%s' to toolkit", tool.name)

    def get_tools(self) -> list[BaseTool]:
        """Get all tools in the toolkit."""
        return self.tools

    def get_tool_by_name(self, name: str) -> BaseTool | None:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def list_tools(self) -> list[str]:
        """List all tool names in the toolkit."""
        return [tool.name for tool in self.tools]

    async def load_from_a2a_servers(
        self,
        server_configs: list[dict[str, Any]],
        adapter: A2ACrewAIAdapter = None,
    ):
        """
        Load tools from multiple A2A servers.

        Args:
            server_configs: List of server configuration dictionaries
            adapter: Optional adapter instance (creates default if not provided)
        """
        if adapter is None:
            adapter = A2ACrewAIAdapter()

        discovery = A2AToolDiscovery(
            server_params=server_configs,
            adapter=adapter,
        )

        discovered_tools = await discovery.discover_and_adapt_tools()

        for tool in discovered_tools:
            self.add_tool(tool)

        logger.info(
            "Loaded %d tools from %d A2A servers",
            len(discovered_tools),
            len(server_configs),
        )
