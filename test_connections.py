#!/usr/bin/env python3
"""Test script to verify all Azure connections are working"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test that Azure configuration loads correctly"""
    print("\n1️⃣ Testing Configuration Loading...")
    print("-" * 50)
    try:
        from rag_attack.utils.config import load_azure_config
        config = load_azure_config()

        print("✅ Configuration loaded successfully!")
        print(f"   - Azure OpenAI Endpoint: {config['openai_endpoint']}")
        print(f"   - Azure Search Endpoint: {config['search_endpoint']}")
        print(f"   - SQL Server: {config['sql_server']}")
        print(f"   - API Base URL: {config['api_base_url']}")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False


def test_openai_connection():
    """Test Azure OpenAI connection"""
    print("\n2️⃣ Testing Azure OpenAI Connection...")
    print("-" * 50)
    try:
        from rag_attack.utils.config import get_openai_client

        client, deployment = get_openai_client()

        # Test with a simple completion
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say 'Hello, Azure OpenAI is working!'"}],
            max_tokens=20
        )

        print(f"✅ Azure OpenAI connection successful!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Azure OpenAI: {e}")
        return False


def test_azure_search_connection():
    """Test Azure Cognitive Search connection"""
    print("\n3️⃣ Testing Azure Search Connection...")
    print("-" * 50)
    try:
        from rag_attack.utils.config import get_search_client

        client = get_search_client()

        # Test with a simple search
        results = client.search(
            search_text="vélo",
            top=1
        )

        count = 0
        for result in results:
            count += 1
            print(f"✅ Azure Search connection successful!")
            print(f"   Found document: {result.get('title', 'No title')[:50]}...")
            break

        if count == 0:
            print("⚠️ Connection successful but no documents found")
            print("   Run 'make search-upload' in scai_onepoint_rag if needed")

        return True
    except Exception as e:
        print(f"❌ Failed to connect to Azure Search: {e}")
        return False


def test_sql_connection():
    """Test SQL Database connection"""
    print("\n4️⃣ Testing SQL Database Connection...")
    print("-" * 50)
    try:
        from rag_attack.utils.config import get_sql_connection
        import pandas as pd

        conn = get_sql_connection()

        # Test with a simple query
        query = "SELECT TOP 1 name FROM sys.tables"
        df = pd.read_sql(query, conn)

        if len(df) > 0:
            print(f"✅ SQL Database connection successful!")
            print(f"   Found table: {df.iloc[0]['name']}")
        else:
            print("⚠️ Connection successful but no tables found")
            print("   Run 'make db-setup' in scai_onepoint_rag if needed")

        conn.close()
        return True
    except Exception as e:
        print(f"❌ Failed to connect to SQL Database: {e}")
        print(f"   Make sure you have ODBC Driver 18 for SQL Server installed")
        print(f"   On Mac: brew install microsoft/mssql-release/msodbcsql18")
        return False


def test_api_endpoints():
    """Test Azure Function API endpoints"""
    print("\n5️⃣ Testing API Endpoints...")
    print("-" * 50)
    try:
        import requests
        from rag_attack.utils.config import load_azure_config

        config = load_azure_config()
        base_url = config["api_base_url"]

        # Test health endpoint
        health_url = base_url.replace("/api", "") + "/api/health"
        response = requests.get(health_url, timeout=10)

        if response.status_code == 200:
            print(f"✅ API endpoint connection successful!")
            print(f"   Health check: {response.text[:100]}")
        else:
            print(f"⚠️ API responded with status {response.status_code}")
            print("   Run 'make api-deploy' in scai_onepoint_rag if needed")

        return True
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False


def test_tool_functions():
    """Test the actual tool functions"""
    print("\n6️⃣ Testing Tool Functions...")
    print("-" * 50)

    # Test search tool
    try:
        from rag_attack.tools import azure_search_tool

        result = azure_search_tool.invoke({"query": "vélo électrique", "top": 1})
        if "Result 1:" in result or "No results" in result:
            print("✅ Azure Search tool working")
        else:
            print("⚠️ Azure Search tool returned unexpected format")
    except Exception as e:
        print(f"❌ Azure Search tool failed: {e}")

    # Test SQL tool
    try:
        from rag_attack.tools import get_database_schema

        result = get_database_schema.invoke({})
        if "Table:" in result or "Error" in result:
            print("✅ SQL Database schema tool working")
        else:
            print("⚠️ SQL tool returned unexpected format")
    except Exception as e:
        print(f"❌ SQL Database tool failed: {e}")

    # Test API tool
    try:
        from rag_attack.tools import azure_function_api_tool

        result = azure_function_api_tool.invoke({
            "endpoint": "/health",
            "method": "GET"
        })
        if result:
            print("✅ API tool working")
        else:
            print("⚠️ API tool returned empty response")
    except Exception as e:
        print(f"❌ API tool failed: {e}")

    return True


def test_agent_creation():
    """Test that agents can be created"""
    print("\n7️⃣ Testing Agent Creation...")
    print("-" * 50)

    try:
        from rag_attack.agents import SimpleToolAgent
        from rag_attack.tools import azure_search_tool

        agent = SimpleToolAgent(
            tools=[azure_search_tool],
            system_prompt="Test agent"
        )

        print("✅ Simple agent created successfully")
    except Exception as e:
        print(f"❌ Failed to create simple agent: {e}")

    try:
        from rag_attack.agents import ReActAgent
        from rag_attack.tools import azure_search_tool, sql_query_tool

        agent = ReActAgent(
            tools=[azure_search_tool, sql_query_tool],
            max_iterations=3
        )

        print("✅ ReAct agent created successfully")
    except Exception as e:
        print(f"❌ Failed to create ReAct agent: {e}")

    try:
        from rag_attack.planners import HierarchicalPlanner
        from rag_attack.tools import azure_search_tool

        planner = HierarchicalPlanner(
            tools=[azure_search_tool],
            max_steps=5
        )

        print("✅ Hierarchical planner created successfully")
    except Exception as e:
        print(f"❌ Failed to create planner: {e}")

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("🔬 TESTING AZURE CONNECTIONS AND TOOLS")
    print("=" * 60)

    results = []

    # Run tests in order
    results.append(("Configuration", test_config_loading()))

    if results[0][1]:  # Only continue if config loaded
        results.append(("Azure OpenAI", test_openai_connection()))
        results.append(("Azure Search", test_azure_search_connection()))
        results.append(("SQL Database", test_sql_connection()))
        results.append(("API Endpoints", test_api_endpoints()))
        results.append(("Tool Functions", test_tool_functions()))
        results.append(("Agent Creation", test_agent_creation()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    total_success = sum(1 for _, s in results if s)
    print(f"\nTotal: {total_success}/{len(results)} tests passed")

    if total_success < len(results):
        print("\n⚠️ Some tests failed. Please check:")
        print("1. Azure resources are deployed: cd scai_onepoint_rag && make setup-all")
        print("2. Required packages are installed: pip install -r requirements.txt")
        print("3. ODBC drivers are installed for SQL access")
    else:
        print("\n🎉 All tests passed! Your environment is ready.")


if __name__ == "__main__":
    main()