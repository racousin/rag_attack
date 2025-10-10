"""RAG Attack - Pedagogic guide for Agentic RAG implementations"""

__version__ = "0.2.0"

# Import main components
from .agents import SimpleToolAgent, create_llm

# Configuration management
from .tools import set_config, get_config

# Clean wrapper functions (RECOMMENDED)
from .tools import (
    search_documents,
    vector_search_documents,
    hybrid_search_documents,
    execute_sql_query,
    get_schema,
    get_table_info,
    call_azure_api,
    search_via_api,
    get_crm_opportunities,
    get_weather,
    search_web,
)

# Original tools (for backward compatibility)
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
    # Configuration
    "set_config",
    "get_config",
    # Clean wrapper functions (RECOMMENDED)
    "search_documents",
    "vector_search_documents",
    "hybrid_search_documents",
    "execute_sql_query",
    "get_schema",
    "get_table_info",
    "call_azure_api",
    "search_via_api",
    "get_crm_opportunities",
    "get_weather",
    "search_web",
    # Original tools (for backward compatibility)
    "azure_search_tool",
    "azure_vector_search_tool",
    "create_hybrid_search_tool",
    "sql_query_tool",
    "get_database_schema",
    "sql_table_info",
    "create_sql_agent_tools",
    "azure_function_api_tool",
    "search_api_tool",
    "crm_opportunities_tool",
    "weather_api_tool",
    "web_search_tool",
    "create_api_tools",
]