"""Configuration management for Azure services"""
import os
import sys
import json
from typing import Dict, Any
from pathlib import Path


def load_azure_config() -> Dict[str, Any]:
    """Load Azure configuration from credentials file or scai_onepoint_rag config

    Priority:
    1. Load from credentials/azure_credentials.json (for students)
    2. Fall back to scai_onepoint_rag config (for development)
    """
    project_root = Path(__file__).parent.parent.parent

    # Try loading from credentials file first (student mode)
    credentials_file = project_root  / "azure_credentials.json"
    if credentials_file.exists():
        try:
            with open(credentials_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Error loading credentials file: {e}")

    # Fall back to scai_onepoint_rag config (development mode)
    try:
        scai_path = project_root / "scai_onepoint_rag"
        sys.path.insert(0, str(scai_path))

        from config.azure_config import config
        return config
    except ImportError:
        raise ImportError(
            "Azure configuration not found. Please either:\n"
            "  1. Place azure_credentials.json in credentials/ directory, OR\n"
            "  2. Run 'make config' in scai_onepoint_rag directory"
        )


def get_openai_client():
    """Initialize Azure OpenAI client"""
    from openai import AzureOpenAI

    config = load_azure_config()

    client = AzureOpenAI(
        api_key=config["openai_key"],
        api_version="2024-02-01",
        azure_endpoint=config["openai_endpoint"]
    )

    return client, config["chat_deployment"]


def get_search_client(index_name: str = "velocorp-documents"):
    """Initialize Azure Cognitive Search client"""
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential

    config = load_azure_config()

    search_client = SearchClient(
        endpoint=config["search_endpoint"],
        index_name=index_name,
        credential=AzureKeyCredential(config["search_key"])
    )

    return search_client


def get_sql_connection():
    """Get SQL database connection string"""
    import pyodbc

    config = load_azure_config()

    connection_string = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={config['sql_server']};"
        f"Database={config['sql_database']};"
        f"Uid={config['sql_username']};"
        f"Pwd={config['sql_password']};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )

    return pyodbc.connect(connection_string)