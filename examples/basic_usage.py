#!/usr/bin/env python3
"""
Basic usage example of CrewAI A2A Adapter

This example demonstrates how to:
1. Connect to an A2A server
2. Discover and adapt tools
3. Create a CrewAI agent with A2A tools
4. Execute tasks
"""

import asyncio
import logging
from crewai import Agent, Crew, Task, Process

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import A2ACrewAIAdapter, CrewAIToolkit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Main execution function."""
    
    # Example A2A server configuration
    # Replace with your actual A2A server details
    server_configs = [
        {
            "url": "https://example-a2a-server.com",
            "headers": {
                "Authorization": "Bearer YOUR_API_TOKEN"
            }
        }
    ]
    
    print("🚀 Starting CrewAI with A2A Tools Example")
    print("-" * 50)
    
    # Create toolkit and load tools from A2A servers
    print("📡 Connecting to A2A servers and discovering tools...")
    toolkit = CrewAIToolkit()
    
    try:
        await toolkit.load_from_a2a_servers(server_configs)
        print(f"✅ Successfully loaded {len(toolkit.get_tools())} tools")
        
        # List available tools
        print("\n📋 Available tools:")
        for tool_name in toolkit.list_tools():
            print(f"  - {tool_name}")
        
    except Exception as e:
        print(f"❌ Failed to load tools: {e}")
        print("Make sure to configure a valid A2A server URL and credentials")
        return
    
    print("\n" + "-" * 50)
    
    # Create a research agent with A2A tools
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Conduct thorough research and provide detailed analysis',
        backstory="""You are an experienced research analyst with expertise in 
        gathering information from various sources and synthesizing it into 
        actionable insights.""",
        tools=toolkit.get_tools(),
        verbose=True
    )
    
    # Create a writer agent (without tools for this example)
    writer = Agent(
        role='Content Writer',
        goal='Create clear and engaging content based on research',
        backstory="""You are a skilled content writer who excels at transforming 
        complex research into easy-to-understand content.""",
        verbose=True
    )
    
    # Define tasks
    research_task = Task(
        description="""Research the current state of artificial intelligence in healthcare. 
        Focus on recent developments, key players, and future predictions.""",
        agent=researcher,
        expected_output="A comprehensive research report with key findings"
    )
    
    writing_task = Task(
        description="""Based on the research findings, write a concise executive summary 
        (maximum 500 words) highlighting the most important points about AI in healthcare.""",
        agent=writer,
        expected_output="An executive summary document"
    )
    
    # Create and configure crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,  # Tasks will be executed sequentially
        verbose=True
    )
    
    # Execute the crew
    print("🎬 Starting crew execution...")
    print("-" * 50)
    
    try:
        result = crew.kickoff()
        
        print("\n" + "=" * 50)
        print("✅ CREW EXECUTION COMPLETED")
        print("=" * 50)
        print("\n📝 Final Result:")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Crew execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())