"""Tools for agentic RAG"""

from .azure_search_tool import (
    azure_search_tool,
    azure_vector_search_tool,
    create_hybrid_search_tool
)
from .sql_tool import (
    sql_query_tool,
    get_database_schema,
    sql_table_info,
    create_sql_agent_tools
)
from .api_tool import (
    azure_function_api_tool,
    search_api_tool,
    crm_opportunities_tool,
    weather_api_tool,
    web_search_tool,
    create_api_tools
)

__all__ = [
    # Search tools
    "azure_search_tool",
    "azure_vector_search_tool",
    "create_hybrid_search_tool",
    # SQL tools
    "sql_query_tool",
    "get_database_schema",
    "sql_table_info",
    "create_sql_agent_tools",
    # API tools
    "azure_function_api_tool",
    "search_api_tool",
    "crm_opportunities_tool",
    "weather_api_tool",
    "web_search_tool",
    "create_api_tools"
]