#!/usr/bin/env python3
"""Test and validate Azure setup for the agentic RAG system"""

import os
import sys
from pathlib import Path


def check_azure_deployment():
    """Check if Azure resources are deployed and provide guidance"""

    print("\n🔍 CHECKING AZURE DEPLOYMENT STATUS")
    print("=" * 60)

    # Check if config exists
    config_path = Path(__file__).parent.parent.parent / "scai_onepoint_rag" / "config" / "azure_config.py"

    if not config_path.exists():
        print("❌ Configuration file not found!")
        print("\n📋 TO FIX THIS:")
        print("1. Navigate to scai_onepoint_rag directory:")
        print("   cd scai_onepoint_rag")
        print("\n2. Deploy Azure resources:")
        print("   make setup-all")
        print("\n3. Wait for deployment to complete (10-15 minutes)")
        print("\n4. Generate config:")
        print("   make config")
        return False

    # Try to load config
    try:
        sys.path.insert(0, str(config_path.parent.parent))
        from config.azure_config import config

        print("✅ Configuration file found")
        print("\nChecking credentials...")

        # Check each service
        issues = []

        # Check OpenAI
        if not config.get("openai_key") or config["openai_key"] == "":
            issues.append("OpenAI API key is missing")
        elif len(config["openai_key"]) < 30:
            issues.append("OpenAI API key looks invalid (too short)")

        # Check Search
        if not config.get("search_key") or config["search_key"] == "":
            issues.append("Search API key is missing")

        # Check SQL
        if not config.get("sql_password") or config["sql_password"] == "":
            issues.append("SQL password is missing")

        if issues:
            print("\n⚠️ CONFIGURATION ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")

            print("\n📋 TO FIX THIS:")
            print("1. Check if resources are deployed:")
            print("   cd scai_onepoint_rag && terraform -chdir=terraform show")
            print("\n2. If not deployed, run:")
            print("   cd scai_onepoint_rag && make setup-all")
            print("\n3. If deployed but config is wrong, regenerate:")
            print("   cd scai_onepoint_rag && make config")
            return False

        print("✅ All credentials appear to be present")
        return True

    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_minimal_setup():
    """Test minimal setup without actual Azure resources"""

    print("\n🧪 TESTING MINIMAL SETUP (No Azure Required)")
    print("=" * 60)

    # Test imports
    try:
        from rag_attack.agents import SimpleToolAgent, ReActAgent
        from rag_attack.planners import HierarchicalPlanner
        print("✅ Agent imports successful")
    except ImportError as e:
        print(f"❌ Failed to import agents: {e}")
        return False

    # Test tool creation (without Azure)
    try:
        from langchain_core.tools import tool

        @tool
        def dummy_search(query: str) -> str:
            """Dummy search tool for testing"""
            return f"Mock search result for: {query}"

        # Create agents with dummy tools
        agent = SimpleToolAgent(
            tools=[dummy_search],
            system_prompt="Test agent"
        )

        print("✅ Agent creation with mock tools successful")

        # Test agent execution with mock
        result = agent.invoke("test query")
        print(f"✅ Agent execution successful: {result[:50]}...")

    except Exception as e:
        print(f"❌ Failed to create/run agent: {e}")
        return False

    return True


def suggest_demo_mode():
    """Suggest how to run in demo mode without Azure"""

    print("\n💡 DEMO MODE SETUP")
    print("=" * 60)
    print("""
You can still explore the agentic RAG concepts without Azure resources!

1. Use mock tools for learning:
   - Create dummy search functions
   - Use in-memory data instead of Azure Search
   - Simulate API responses

2. Focus on architecture concepts:
   - Agent workflows with LangGraph
   - ReAct reasoning patterns
   - Hierarchical planning

3. Example mock setup:
""")

    print("""```python
from langchain_core.tools import tool
from rag_attack.agents import SimpleToolAgent

@tool
def mock_search(query: str) -> str:
    \"\"\"Mock search for demo purposes\"\"\"
    mock_data = {
        "vélo": "VéloCorp propose 3 modèles: Urban, Sport, Elite",
        "prix": "Prix de 1500€ à 4500€",
        "garantie": "Garantie 2 ans sur tous les modèles"
    }
    for key, value in mock_data.items():
        if key in query.lower():
            return value
    return "No data found"

# Create agent with mock tool
agent = SimpleToolAgent(tools=[mock_search])
response = agent.invoke("Quels sont les prix des vélos?")
print(response)
```""")


def main():
    """Main test function"""

    print("=" * 60)
    print("🔧 AZURE SETUP DIAGNOSTIC TOOL")
    print("=" * 60)

    # Check Azure deployment
    azure_ok = check_azure_deployment()

    # Test minimal setup
    minimal_ok = test_minimal_setup()

    # Provide recommendations
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    if azure_ok:
        print("✅ Azure resources are configured")
        print("You can run the full notebook with real data!")
    else:
        print("⚠️ Azure resources not fully configured")

        if minimal_ok:
            print("✅ But the package is working correctly!")
            print("\nYou have two options:")
            print("1. Deploy Azure resources (recommended for full experience)")
            print("2. Use demo mode with mock data (good for learning)")

            suggest_demo_mode()
        else:
            print("❌ Package setup also has issues")
            print("\nPlease:")
            print("1. Install dependencies: poetry install")
            print("2. Check Python version: python --version (should be 3.8+)")


if __name__ == "__main__":
    main()