"""Tools for agentic RAG - VéloCorp"""

# Configuration management
from ..utils.config import set_config, get_config

# 6 consolidated tools
from .crm_tool import get_crm
from .erp_tool import get_erp
from .rag_tool import get_document_rag
from .internet_tool import get_internet_search
from .writer_tool import write_file
from .mail_tool import send_mail

__all__ = [
    # Configuration
    "set_config",
    "get_config",
    # Tools (6 total)
    "get_crm",           # CRM data (opportunities, prospects, sales_reps, analytics)
    "get_erp",           # SQL database (products, customers, orders, etc.)
    "get_document_rag",  # Document search (keyword, vector, hybrid)
    "get_internet_search",  # Web search
    "write_file",        # Report generation (text, excel)
    "send_mail",         # Send emails via SMTP
]
