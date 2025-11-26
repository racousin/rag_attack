"""RAG Attack - Pedagogic package for Agentic RAG implementations"""

__version__ = "0.5.0"

# Agents
from .agents import SimpleAgent, ReflectionAgent, create_llm, display_graph, VerboseLevel

# Configuration
from .tools import set_config, get_config

# 6 Tools for agents
from .tools import (
    get_crm,             # CRM data (opportunities, prospects, sales_reps, analytics)
    get_erp,             # SQL database (products, customers, orders, etc.)
    get_document_rag,    # Document search (keyword, vector, hybrid)
    get_internet_search, # Web search
    write_file,          # Report generation (text, excel)
    send_mail,           # Send emails via SMTP
)

__all__ = [
    # Agents
    "SimpleAgent",
    "ReflectionAgent",
    "create_llm",
    "display_graph",
    "VerboseLevel",
    # Configuration
    "set_config",
    "get_config",
    # Tools (6 total)
    "get_crm",
    "get_erp",
    "get_document_rag",
    "get_internet_search",
    "write_file",
    "send_mail",
]
