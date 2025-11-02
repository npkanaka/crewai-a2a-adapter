#!/usr/bin/env python3
"""
Advanced integration example for CrewAI A2A Adapter

This example demonstrates:
1. Connecting to multiple A2A servers
2. Custom tool filtering and selection
3. Hierarchical crew process with manager agent
4. Error handling and retry logic
5. Custom timeout configuration
"""

import asyncio
import logging
from datetime import timedelta
from typing import List, Optional
from crewai import Agent, Crew, Task, Process

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    A2ASession,
    A2ACrewAIAdapter,
    CrewAIToolkit,
    StreamingCrew,
    SimpleStreamingCallback,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedToolkit(CrewAIToolkit):
    """Extended toolkit with filtering capabilities."""
    
    def filter_tools_by_name(self, patterns: List[str]):
        """Filter tools by name patterns."""
        filtered = []
        for tool in self.tools:
            for pattern in patterns:
                if pattern.lower() in tool.name.lower():
                    filtered.append(tool)
                    break
        self.tools = filtered
        logger.info(f"Filtered to {len(self.tools)} tools matching patterns: {patterns}")
    
    def get_tool_descriptions(self) -> dict:
        """Get a dictionary of tool names and their descriptions."""
        return {tool.name: tool.description for tool in self.tools}


async def connect_with_retry(
    server_config: dict,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> Optional[List]:
    """
    Connect to an A2A server with retry logic.
    
    Args:
        server_config: Server configuration dictionary
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        List of discovered tools or None if connection fails
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting connection to {server_config['url']} (attempt {attempt + 1}/{max_retries})")
            
            # Custom timeout for this server
            timeout = server_config.get('timeout', timedelta(seconds=30))
            
            session = A2ASession(
                server_url=server_config['url'],
                headers=server_config.get('headers'),
                timeout=timeout
            )
            
            await session.connect()
            tools = session.tools
            await session.disconnect()
            
            logger.info(f"Successfully connected and discovered {len(tools)} tools")
            return tools
            
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                return None


async def setup_multi_server_toolkit() -> AdvancedToolkit:
    """
    Set up toolkit with tools from multiple A2A servers.
    
    Returns:
        Configured toolkit with tools from all available servers
    """
    # Configure multiple A2A servers with different settings
    server_configs = [
        {
            "url": "https://research-agent.example.com",
            "headers": {"Authorization": "Bearer RESEARCH_TOKEN"},
            "timeout": timedelta(seconds=60),
            "name": "Research Agent"
        },
        {
            "url": "https://data-agent.example.com",
            "headers": {"Authorization": "Bearer DATA_TOKEN"},
            "timeout": timedelta(seconds=120),
            "name": "Data Analysis Agent"
        },
        {
            "url": "https://web-agent.example.com",
            "headers": {"API-Key": "WEB_API_KEY"},
            "timeout": timedelta(seconds=30),
            "name": "Web Search Agent"
        }
    ]
    
    toolkit = AdvancedToolkit()
    adapter = A2ACrewAIAdapter()
    
    # Connect to each server with retry logic
    for config in server_configs:
        print(f"\n📡 Connecting to {config['name']}...")
        
        tools = await connect_with_retry(config)
        if tools:
            # Adapt each tool
            for tool in tools:
                try:
                    adapted_tool = await adapter.async_adapt(tool, config)
                    toolkit.add_tool(adapted_tool)
                    print(f"  ✅ Added tool: {tool.name}")
                except Exception as e:
                    print(f"  ⚠️  Failed to adapt tool {tool.name}: {e}")
        else:
            print(f"  ❌ Could not connect to {config['name']}")
    
    return toolkit


async def main():
    """Main execution with advanced features."""
    
    print("🚀 Advanced CrewAI A2A Integration Example")
    print("=" * 60)
    
    # Set up multi-server toolkit
    print("\n🔧 Setting up multi-server toolkit...")
    toolkit = await setup_multi_server_toolkit()
    
    if not toolkit.get_tools():
        print("\n⚠️  No tools available. Using mock configuration for demonstration.")
        # Continue with example even without real tools
    else:
        print(f"\n✅ Total tools available: {len(toolkit.get_tools())}")
        
        # Optionally filter tools
        # toolkit.filter_tools_by_name(["search", "analyze", "research"])
        
        # Display tool descriptions
        print("\n📋 Available tools and descriptions:")
        for name, desc in toolkit.get_tool_descriptions().items():
            print(f"  • {name}: {desc[:80]}...")
    
    print("\n" + "-" * 60)
    
    # Create specialized agents with specific tools
    researcher = Agent(
        role='Lead Researcher',
        goal='Conduct comprehensive research using multiple data sources',
        backstory="""You are the lead researcher with access to various research 
        tools and databases. You excel at finding and synthesizing information 
        from multiple sources.""",
        tools=[t for t in toolkit.get_tools() if 'research' in t.name.lower()]
        if toolkit.get_tools() else [],
        verbose=True,
        max_iter=5,  # Limit iterations to prevent infinite loops
        memory=True  # Enable memory for better context retention
    )
    
    analyst = Agent(
        role='Senior Data Analyst',
        goal='Analyze data and identify patterns and insights',
        backstory="""You are a senior data analyst specializing in complex data 
        analysis and pattern recognition. You work closely with the research team.""",
        tools=[t for t in toolkit.get_tools() if 'data' in t.name.lower() or 'analyze' in t.name.lower()]
        if toolkit.get_tools() else [],
        verbose=True,
        max_iter=5,
        memory=True
    )
    
    strategist = Agent(
        role='Strategy Director',
        goal='Develop actionable strategies based on research and analysis',
        backstory="""You are the strategy director responsible for translating 
        research and analysis into concrete business strategies.""",
        tools=[],  # Strategist doesn't need tools, just synthesizes information
        verbose=True,
        memory=True
    )
    
    # Manager agent for hierarchical process
    manager = Agent(
        role='Project Manager',
        goal='Coordinate the team to deliver comprehensive insights',
        backstory="""You are an experienced project manager who ensures all team 
        members work together effectively to deliver the best results.""",
        verbose=True,
        memory=True,
        allow_delegation=True  # Allow manager to delegate tasks
    )
    
    # Define interconnected tasks
    research_task = Task(
        description="""Research the impact of artificial intelligence on job markets 
        in the next 5 years. Focus on:
        1. Industries most likely to be affected
        2. New job categories being created
        3. Required skills for the future workforce
        
        Provide concrete data and examples.""",
        agent=researcher,
        expected_output="Comprehensive research report with data and sources"
    )
    
    analysis_task = Task(
        description="""Analyze the research findings to identify:
        1. Key trends and patterns
        2. Risk factors and opportunities
        3. Regional differences in AI adoption
        4. Timeline of expected changes
        
        Use statistical analysis where possible.""",
        agent=analyst,
        expected_output="Detailed analysis with visualizations and statistics"
    )
    
    strategy_task = Task(
        description="""Based on the research and analysis, develop:
        1. Strategic recommendations for businesses
        2. Policy recommendations for governments
        3. Educational strategies for workforce preparation
        4. Risk mitigation strategies
        
        Each recommendation should be actionable with clear steps.""",
        agent=strategist,
        expected_output="Strategic plan with actionable recommendations"
    )
    
    # Create hierarchical crew with manager
    crew = Crew(
        agents=[researcher, analyst, strategist],
        tasks=[research_task, analysis_task, strategy_task],
        manager_agent=manager,
        process=Process.hierarchical,  # Use hierarchical process with manager
        verbose=True
    )
    
    # Set up streaming for real-time updates
    print("\n🎬 Starting hierarchical crew execution...")
    print("-" * 60)
    
    callback = SimpleStreamingCallback(prefix="[HIERARCHY]")
    streaming_crew = StreamingCrew(crew, callback)
    
    try:
        # Execute with timeout
        result = await asyncio.wait_for(
            streaming_crew.kickoff_async(),
            timeout=600  # 10 minute timeout for entire execution
        )
        
        print("\n" + "=" * 60)
        print("✅ EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\n📊 Final Results:")
        print(result)
        
    except asyncio.TimeoutError:
        print("\n⏰ Execution timed out after 10 minutes")
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
    except Exception as e:
        logger.error(f"Execution failed with error: {e}", exc_info=True)
        print(f"\n❌ Execution failed: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)