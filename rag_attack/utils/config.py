"""Configuration management for Azure services"""
from typing import Dict, Any, Optional


# Global configuration storage
_global_config: Optional[Dict[str, Any]] = None


def set_config(config: Dict[str, Any]) -> None:
    """Set the global Azure configuration

    Args:
        config: Dictionary containing Azure credentials and endpoints
            Required keys:
            - openai_key: Azure OpenAI API key
            - openai_endpoint: Azure OpenAI endpoint URL
            - chat_deployment: Azure OpenAI chat deployment name
            - search_endpoint: Azure Cognitive Search endpoint
            - search_key: Azure Cognitive Search API key
            Optional keys:
            - sql_server: SQL server address
            - sql_database: SQL database name
            - sql_username: SQL username
            - sql_password: SQL password
            - smtp_server: SMTP server (default: smtp.gmail.com)
            - smtp_port: SMTP port (default: 465)
            - mail_sender: Sender email address (default: raphaelcousin.education@gmail.com)
            - mail_password: Gmail App Password for authentication
    """
    global _global_config
    _global_config = config


def get_config() -> Dict[str, Any]:
    """Get the global Azure configuration

    Returns:
        The global configuration dictionary

    Raises:
        RuntimeError: If configuration has not been set
    """
    if _global_config is None:
        raise RuntimeError(
            "Configuration not set. Please call set_config() with your Azure credentials first.\n"
            "Example:\n"
            "  from rag_attack.utils.config import set_config\n"
            "  set_config({\n"
            "      'openai_key': 'your-key',\n"
            "      'openai_endpoint': 'https://your-endpoint.openai.azure.com/',\n"
            "      'chat_deployment': 'your-deployment',\n"
            "      'search_endpoint': 'https://your-search.search.windows.net',\n"
            "      'search_key': 'your-search-key'\n"
            "  })"
        )
    return _global_config


def get_openai_client():
    """Initialize Azure OpenAI client"""
    from openai import AzureOpenAI

    config = get_config()

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

    config = get_config()

    search_client = SearchClient(
        endpoint=config["search_endpoint"],
        index_name=index_name,
        credential=AzureKeyCredential(config["search_key"])
    )

    return search_client


def get_sql_connection():
    """Get SQL database connection string"""
    import pyodbc

    config = get_config()

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