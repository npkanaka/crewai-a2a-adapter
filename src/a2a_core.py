import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, AsyncGenerator, List, Optional
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TextPart,
    Message,
    AgentCard,
    GetTaskRequest,
    TaskQueryParams,
    Task,
    TaskState,
    Part,
)

logger = logging.getLogger(__name__)


@dataclass
class A2ATool:
    name: str
    description: str
    input_schema: dict
    output_schema: Optional[dict] = None


@dataclass
class A2ACallToolResult:
    success: bool
    result: Any
    error: Optional[str] = None


class A2ASession:

    def __init__(
        self,
        server_url: str,
        headers: Optional[dict] = None,
        timeout: Optional[timedelta] = None,
    ):
        self.server_url = server_url
        self.headers = headers or {}
        self.timeout = timeout or timedelta(seconds=180)
        self.httpx_client = None
        self.card_resolver = None
        self.client = None
        self.agent_card = None
        self.tools: List[A2ATool] = []

    async def connect(self):
        try:
            timeout_config = httpx.Timeout(self.timeout.total_seconds())
            self.httpx_client = httpx.AsyncClient(timeout=timeout_config, headers=self.headers)

            self.card_resolver = A2ACardResolver(
                self.httpx_client, self.server_url, "/.well-known/agent.json"
            )
            self.agent_card = await self.card_resolver.get_agent_card()

            self.client = A2AClient(httpx_client=self.httpx_client, agent_card=self.agent_card)

            await self._discover_capabilities()

            logger.info(
                "Connected to A2A server at %s, found %d tools",
                self.server_url,
                len(self.tools),
            )

        except Exception:
            logger.exception("Failed to connect to A2A server %s", self.server_url)
            await self.disconnect()
            raise

    async def _discover_capabilities(self):
        self.tools = []

        if not self.agent_card:
            return

        if hasattr(self.agent_card, "skills") and self.agent_card.skills:
            for skill in self.agent_card.skills:
                skill_name = getattr(skill, "id", getattr(skill, "name", "unknown"))
                skill_description = getattr(skill, "description", "")

                input_schema = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": f"Input for {skill_name}",
                        }
                    },
                }

                self.tools.append(
                    A2ATool(
                        name=skill_name,
                        description=skill_description,
                        input_schema=input_schema,
                        output_schema=None,
                    )
                )

        elif hasattr(self.agent_card, "capabilities"):
            capabilities = self.agent_card.capabilities
            if hasattr(capabilities, "tools") and capabilities.tools:
                for tool_spec in capabilities.tools:
                    self.tools.append(
                        A2ATool(
                            name=getattr(tool_spec, "name", "unknown"),
                            description=getattr(tool_spec, "description", ""),
                            input_schema=getattr(tool_spec, "input_schema", {}),
                            output_schema=getattr(tool_spec, "output_schema", None),
                        )
                    )

    async def call_tool(
        self, tool_name: str, arguments: Optional[dict] = None
    ) -> A2ACallToolResult:
        if not self.client:
            logger.error("A2A client not connected.")
            return A2ACallToolResult(success=False, result=None, error="Client not connected")

        logger.info(
            "Executing A2A tool '%s' with arguments: %s", tool_name, list((arguments or {}).keys())
        )

        try:
            async with asyncio.timeout(self.timeout.total_seconds()):
                if arguments and "input" in arguments:
                    message_text = str(arguments["input"])
                elif arguments:
                    message_text = json.dumps(arguments, default=str)
                else:
                    message_text = ""

                metadata = {}
                if appstack_id := os.getenv("APPSTACK_ID"):
                    metadata["appstackID"] = appstack_id

                send_request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        message={
                            "role": "user",
                            "parts": [Part(root=TextPart(text=message_text))],
                            "messageId": str(uuid4()),
                            "metadata": metadata,
                        },
                        configuration={
                            "blocking": False,
                            "acceptedOutputModes": ["text/plain", "text"],
                        },
                    ),
                )

                http_kwargs = {"headers": self.headers} if self.headers else {}
                initial_response = await self.client.send_message(
                    send_request, http_kwargs=http_kwargs
                )

                if not (
                    hasattr(initial_response, "root") and hasattr(initial_response.root, "result")
                ):
                    return A2ACallToolResult(
                        success=False, result=None, error="Invalid response from A2A server"
                    )

                task = initial_response.root.result

                if isinstance(task, Message) and task.parts:
                    text_part = getattr(task.parts[0], "root", task.parts[0])
                    if hasattr(text_part, "text") and text_part.text:
                        return A2ACallToolResult(success=True, result=text_part.text)

                if not isinstance(task, Task):
                    return A2ACallToolResult(
                        success=False, result=None, error=f"Unexpected response type: {type(task)}"
                    )

                if task.status.state in [TaskState.completed, TaskState.failed]:
                    return await self._get_task_result(task)

                return await self._poll_task(task.id, http_kwargs)

        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after {self.timeout.total_seconds()} seconds"
            logger.error(error_msg)
            return A2ACallToolResult(success=False, result=None, error=error_msg)
        except Exception as e:
            logger.exception("Error executing A2A tool '%s'", tool_name)
            return A2ACallToolResult(success=False, result=None, error=str(e))

    async def _poll_task(self, task_id: str, http_kwargs: dict) -> A2ACallToolResult:
        base_interval = 0.2
        max_interval = 3.0
        current_interval = base_interval

        while True:
            await asyncio.sleep(current_interval)

            get_task_request = GetTaskRequest(
                id=str(uuid4()),
                params=TaskQueryParams(id=task_id, historyLength=10),
            )
            get_task_response = await self.client.get_task(
                get_task_request, http_kwargs=http_kwargs
            )

            if not (
                hasattr(get_task_response, "root") and hasattr(get_task_response.root, "result")
            ):
                current_interval = min(current_interval * 1.5, max_interval)
                continue

            task = get_task_response.root.result

            if task.status.state in [TaskState.completed, TaskState.failed]:
                return await self._get_task_result(task)

            current_interval = min(current_interval * 1.5, max_interval)

    async def _get_task_result(self, task: Task) -> A2ACallToolResult:
        if task.status.state == TaskState.failed:
            error_message = "Task failed"
            if task.history and task.history[-1].parts:
                text_part = getattr(task.history[-1].parts[0], "root", task.history[-1].parts[0])
                if hasattr(text_part, "text") and text_part.text:
                    error_message = text_part.text
            return A2ACallToolResult(success=False, result=None, error=error_message)

        if task.status.state == TaskState.completed:
            final_result = "Task completed"
            if task.history and task.history[-1].parts:
                text_part = getattr(task.history[-1].parts[0], "root", task.history[-1].parts[0])
                if hasattr(text_part, "text") and text_part.text:
                    final_result = text_part.text
            return A2ACallToolResult(success=True, result=final_result)

        return A2ACallToolResult(
            success=False,
            result=None,
            error=f"Unexpected task state: {task.status.state}",
        )

    async def disconnect(self):
        if self.httpx_client:
            await self.httpx_client.aclose()


class ToolAdapter(ABC):
    @abstractmethod
    async def async_adapt(
        self,
        a2a_tool: A2ATool,
        server_params: dict[str, Any],
    ) -> Any:
        raise NotImplementedError


@asynccontextmanager
async def a2a_session(
    server_params: dict[str, Any],
    timeout: Optional[timedelta] = None,
) -> AsyncGenerator[tuple[A2ASession, list[A2ATool]], None]:
    session = A2ASession(
        server_url=server_params["url"],
        headers=server_params.get("headers"),
        timeout=timeout or timedelta(seconds=180),
    )

    try:
        await session.connect()
        yield session, session.tools
    finally:
        await session.disconnect()


class A2AToolDiscovery:
    def __init__(
        self,
        server_params: list[dict[str, Any]],
        adapter: ToolAdapter,
        connect_timeout: timedelta = timedelta(seconds=180),
    ):
        self.server_params = server_params
        self.adapter = adapter
        self.connect_timeout = connect_timeout

    async def discover_and_adapt_tools(self) -> list[Any]:
        adapted_tools = []

        for params in self.server_params:
            try:
                async with a2a_session(params, self.connect_timeout) as (session, tools):
                    for tool in tools:
                        adapted_tool = await self.adapter.async_adapt(tool, params)
                        adapted_tools.append(adapted_tool)
                        logger.info("Adapted tool '%s' from %s", tool.name, params.get("url"))
            except Exception:
                logger.exception("Failed to connect to A2A server %s", params.get("url"))
                continue

        return adapted_tools
