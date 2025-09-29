"""RAG Attack - Pedagogic guide for Agentic RAG implementations"""

__version__ = "0.2.0"

# Import main components
from .agents import SimpleToolAgent, ReActAgent
from .tools import (
    azure_search_tool,
    azure_vector_search_tool,
    sql_query_tool,
    azure_function_api_tool,
    crm_opportunities_tool
)
from .planners import HierarchicalPlanner
from .utils import load_azure_config, get_openai_client

__all__ = [
    # Agents
    "SimpleToolAgent",
    "ReActAgent",
    # Tools
    "azure_search_tool",
    "azure_vector_search_tool",
    "sql_query_tool",
    "azure_function_api_tool",
    "crm_opportunities_tool",
    # Planners
    "HierarchicalPlanner",
    # Utils
    "load_azure_config",
    "get_openai_client",
]