"""Tools for agentic RAG"""

# Configuration management - import from centralized location
from ..utils.config import (
    set_config,
    get_config,
)

# Original tool functions (with explicit config parameter)
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

# Clean wrapper functions (use global config, no config parameter needed)
from .azure_search_tool import (
    search_documents,
    vector_search_documents,
    hybrid_search_documents,
)
from .sql_tool import (
    execute_sql_query,
    get_schema,
    get_table_info,
)
from .api_tool import (
    call_azure_api,
    search_via_api,
    get_crm_opportunities,
    get_weather,
    search_web,
)

__all__ = [
    # Configuration
    "set_config",
    "get_config",
    # Clean wrapper functions (RECOMMENDED - use these!)
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
    "create_api_tools"
]