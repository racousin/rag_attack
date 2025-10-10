"""RAG Attack - Pedagogic guide for Agentic RAG implementations"""

__version__ = "0.2.0"

# Import main components
from .agents import SimpleToolAgent, create_llm
from .tools import (
    azure_search_tool,
    azure_vector_search_tool,
    sql_query_tool,
    get_database_schema,
    sql_table_info,
    azure_function_api_tool,
    search_api_tool,
    crm_opportunities_tool,
    weather_api_tool,
    web_search_tool,
    create_api_tools,
    create_sql_agent_tools,
    create_hybrid_search_tool
)

__all__ = [
    # Agents
    "SimpleToolAgent",
    "create_llm",
    # Search Tools
    "azure_search_tool",
    "azure_vector_search_tool",
    "create_hybrid_search_tool",
    # SQL Tools
    "sql_query_tool",
    "get_database_schema",
    "sql_table_info",
    "create_sql_agent_tools",
    # API Tools
    "azure_function_api_tool",
    "search_api_tool",
    "crm_opportunities_tool",
    "weather_api_tool",
    "web_search_tool",
    "create_api_tools",
]