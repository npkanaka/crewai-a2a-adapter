#!/usr/bin/env python3
"""
Streaming example for CrewAI A2A Adapter

This example demonstrates:
1. Real-time streaming of agent thoughts and actions
2. Custom callback implementation
3. Monitoring tool usage and agent interactions
"""

import asyncio
import logging
from datetime import datetime
from crewai import Agent, Crew, Task, Process

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    CrewAIToolkit,
    StreamingCrew,
    StreamingCallback,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RichStreamingCallback(StreamingCallback):
    """
    Custom streaming callback with rich console output.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.tool_usage_count = {}
        self.agent_activity = {}
    
    def _elapsed_time(self) -> str:
        """Get elapsed time since start."""
        elapsed = datetime.now() - self.start_time
        return f"[{elapsed.total_seconds():.1f}s]"
    
    async def on_agent_start(self, agent_name: str, context: str):
        """Called when an agent starts execution."""
        print(f"\n{self._elapsed_time()} 🤖 AGENT START: {agent_name}")
        print(f"    📋 Context: {context}")
        self.agent_activity[agent_name] = {"start": datetime.now(), "tools_used": 0}
    
    async def on_agent_finish(self, agent_name: str, result: str):
        """Called when an agent finishes execution."""
        if agent_name in self.agent_activity:
            duration = datetime.now() - self.agent_activity[agent_name]["start"]
            tools_used = self.agent_activity[agent_name]["tools_used"]
            print(f"\n{self._elapsed_time()} ✅ AGENT COMPLETE: {agent_name}")
            print(f"    ⏱️  Duration: {duration.total_seconds():.1f}s")
            print(f"    🔧 Tools used: {tools_used}")
            print(f"    📊 Result preview: {result[:150]}...")
    
    async def on_agent_thought(self, agent_name: str, thought: str):
        """Called when an agent produces a thought."""
        print(f"\n{self._elapsed_time()} 💭 {agent_name} thinking:")
        # Format thought with indentation
        for line in thought.split('\n'):
            if line.strip():
                print(f"    > {line}")
    
    async def on_tool_start(self, agent_name: str, tool_name: str, tool_input: str):
        """Called when a tool execution starts."""
        print(f"\n{self._elapsed_time()} 🔧 TOOL START")
        print(f"    Agent: {agent_name}")
        print(f"    Tool: {tool_name}")
        print(f"    Input: {tool_input[:100]}...")
        
        # Track tool usage
        if tool_name not in self.tool_usage_count:
            self.tool_usage_count[tool_name] = 0
        self.tool_usage_count[tool_name] += 1
        
        if agent_name in self.agent_activity:
            self.agent_activity[agent_name]["tools_used"] += 1
    
    async def on_tool_finish(self, agent_name: str, tool_name: str, result: str):
        """Called when a tool execution finishes."""
        print(f"{self._elapsed_time()} ✔️  TOOL COMPLETE: {tool_name}")
        if "error" in result.lower():
            print(f"    ⚠️  Result: {result[:100]}...")
        else:
            print(f"    📦 Result: {result[:100]}...")
    
    async def on_agent_delegation(self, from_agent: str, to_agent: str, task: str):
        """Called when an agent delegates work to another agent."""
        print(f"\n{self._elapsed_time()} 🤝 DELEGATION")
        print(f"    From: {from_agent}")
        print(f"    To: {to_agent}")
        print(f"    Task: {task[:100]}...")
    
    async def on_coworker_interaction(
        self, agent_name: str, coworker_name: str, interaction_type: str, content: str
    ):
        """Called when agents interact with each other."""
        print(f"\n{self._elapsed_time()} 💬 AGENT INTERACTION")
        print(f"    {agent_name} → {coworker_name}")
        print(f"    Type: {interaction_type}")
        print(f"    Content: {content[:100]}...")
    
    async def on_crew_finish(self, final_result: str):
        """Called when the entire crew finishes execution."""
        total_duration = datetime.now() - self.start_time
        
        print("\n" + "=" * 60)
        print("🎉 CREW EXECUTION COMPLETE")
        print("=" * 60)
        print(f"\n📊 EXECUTION STATISTICS:")
        print(f"  Total duration: {total_duration.total_seconds():.1f}s")
        print(f"  Agents involved: {len(self.agent_activity)}")
        
        if self.tool_usage_count:
            print(f"\n🔧 TOOL USAGE:")
            for tool, count in self.tool_usage_count.items():
                print(f"  - {tool}: {count} time(s)")
        
        print(f"\n📝 FINAL RESULT:")
        print("-" * 40)
        print(final_result[:500])
        if len(final_result) > 500:
            print(f"... (truncated, {len(final_result)} total characters)")
        print("-" * 40)


async def main():
    """Main execution function with streaming."""
    
    # Example A2A server configuration
    server_configs = [
        {
            "url": "https://example-a2a-server.com",
            "headers": {"Authorization": "Bearer YOUR_API_TOKEN"}
        }
    ]
    
    print("🚀 CrewAI A2A Adapter - Streaming Example")
    print("=" * 60)
    
    # Load A2A tools
    print("\n📡 Loading tools from A2A servers...")
    toolkit = CrewAIToolkit()
    
    try:
        await toolkit.load_from_a2a_servers(server_configs)
        print(f"✅ Loaded {len(toolkit.get_tools())} tools successfully")
    except Exception as e:
        print(f"⚠️  Note: Could not connect to A2A server: {e}")
        print("   Continuing with example agents (no external tools)...")
        # Continue with the example even without tools
    
    # Create agents
    analyst = Agent(
        role='Data Analyst',
        goal='Analyze data and extract meaningful insights',
        backstory="""You are an experienced data analyst with strong analytical 
        skills and attention to detail. You excel at finding patterns and trends 
        in complex datasets.""",
        tools=toolkit.get_tools() if toolkit.get_tools() else [],
        verbose=False  # We'll use streaming instead
    )
    
    strategist = Agent(
        role='Business Strategist',
        goal='Develop strategic recommendations based on analysis',
        backstory="""You are a senior business strategist with years of experience 
        in translating data insights into actionable business strategies.""",
        verbose=False
    )
    
    # Define tasks
    analysis_task = Task(
        description="""Analyze the current trends in renewable energy adoption 
        globally. Identify the top 3 countries leading in renewable energy and 
        their key strategies.""",
        agent=analyst,
        expected_output="Detailed analysis with data points and trends"
    )
    
    strategy_task = Task(
        description="""Based on the renewable energy analysis, develop 3 strategic 
        recommendations for businesses looking to transition to renewable energy 
        sources.""",
        agent=strategist,
        expected_output="Strategic recommendations with implementation steps"
    )
    
    # Create crew
    crew = Crew(
        agents=[analyst, strategist],
        tasks=[analysis_task, strategy_task],
        process=Process.sequential,
        verbose=False  # Disable default verbose since we're using streaming
    )
    
    # Create streaming wrapper with custom callback
    print("\n🎬 Starting crew execution with streaming...")
    print("-" * 60)
    
    callback = RichStreamingCallback()
    streaming_crew = StreamingCrew(crew, callback)
    
    try:
        # Execute with streaming
        result = await streaming_crew.kickoff_async()
        
        # Result is already displayed by the callback
        # Additional processing can be done here if needed
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")