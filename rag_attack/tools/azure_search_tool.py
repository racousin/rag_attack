"""Azure Cognitive Search tool for RAG"""
from typing import List, Dict, Any, Optional, Callable
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Import centralized config management
from ..utils.config import set_config, get_config


class SearchQuery(BaseModel):
    """Input schema for Azure Search tool"""
    query: str = Field(description="The search query to execute")
    top: int = Field(default=5, description="Number of results to return")
    search_fields: Optional[str] = Field(default=None, description="Comma-separated list of fields to search")


@tool
def azure_search_tool(query: str, top: int = 5, search_fields: Optional[str] = None, index_name: str = "documents") -> str:
    """Search documents in Azure Cognitive Search using keyword search (BM25).

    The index contains VéloCorp documents: FAQs, user manuals, emails, call transcripts.

    Available searchable fields:
    - content: Main text content (default if search_fields not specified)
    - filename: Document filename

    Available filter fields (use with filter parameter, not search_fields):
    - type: Document type (e.g., "manuel", "conversation", "autre")
    - created: Creation date
    - document_id: Numeric document ID

    Args:
        query: The search query (keywords to search for)
        top: Number of results to return (default: 5)
        search_fields: Comma-separated searchable fields (e.g., "content", "filename", "content,filename")
        index_name: Name of the search index (default: "documents")

    Returns:
        Formatted search results as a string with title, category, source, and content excerpt

    Examples:
        - azure_search_tool("garantie vélo", top=3) - Search in all fields
        - azure_search_tool("E-City", search_fields="filename") - Search only in filenames
        - azure_search_tool("freins hydrauliques", search_fields="content") - Search only in content
    """
    config = get_config()
    try:
        client = SearchClient(
            endpoint=config["search_endpoint"],
            index_name=index_name,
            credential=AzureKeyCredential(config["search_key"])
        )

        # Prepare search options
        search_options = {
            "top": top,
            "include_total_count": True,
            "search_mode": "any"
        }

        if search_fields:
            # Convert search_fields to list if it's a string
            if isinstance(search_fields, str):
                # Split by comma and strip whitespace
                search_options["search_fields"] = [field.strip() for field in search_fields.split(",")]
            else:
                search_options["search_fields"] = search_fields

        # Execute search
        results = client.search(
            search_text=query,
            **search_options
        )

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            # Extract key fields
            title = result.get("title", "No Title")
            content = result.get("content", "")[:500]  # Truncate content
            category = result.get("category", "Unknown")
            source = result.get("source", "Unknown")

            formatted_results.append(
                f"Result {i}:\n"
                f"  Title: {title}\n"
                f"  Category: {category}\n"
                f"  Source: {source}\n"
                f"  Content: {content}...\n"
            )

        if not formatted_results:
            return f"No results found for query: '{query}'"

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching Azure Cognitive Search: {str(e)}"


class VectorSearchQuery(BaseModel):
    """Input schema for vector search"""
    query: str = Field(description="The search query to vectorize")
    top: int = Field(default=5, description="Number of results to return")
    filter: Optional[str] = Field(default=None, description="OData filter expression")


@tool
def azure_vector_search_tool(query: str, top: int = 5, filter: Optional[str] = None, index_name: str = "documents") -> str:
    """Perform semantic vector search using Azure Cognitive Search with embeddings.

    Uses sentence-transformers/all-MiniLM-L6-v2 (384D) for semantic similarity search.
    Better for conceptual queries, intent-based search, and finding similar content.

    The index contains VéloCorp documents: FAQs, user manuals, emails, call transcripts.

    Args:
        query: The search query to vectorize (natural language question or concept)
        top: Number of results to return (default: 5)
        filter: OData filter expression for additional filtering
                Examples: "type eq 'manuel'", "document_id gt 10"
        index_name: Name of the search index (default: "documents")

    Returns:
        Formatted search results with relevance scores and content excerpts

    When to use:
        - Conceptual queries: "how to maintain my bike"
        - Similar meaning: "battery problems" vs "batterie défaillante"
        - Intent-based: "je veux savoir comment réparer"

    When NOT to use (use azure_search_tool instead):
        - Exact terms: "SKU-ELC-2024"
        - Product names: "E-City"
        - Specific keywords: "garantie 2 ans"
    """
    config = get_config()
    try:
        from langchain.embeddings import HuggingFaceEmbeddings

        # Use the same embedding model as upload_azure_search.py
        embedding_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Generate embedding for the query (returns list of lists, take first)
        query_embedding = embedding_fn.embed_query(query)

        # Get search client
        client = SearchClient(
            endpoint=config["search_endpoint"],
            index_name=index_name,
            credential=AzureKeyCredential(config["search_key"])
        )

        # Prepare vector search
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top,
            fields="embedding"  # Matches the field name in create_search_index.py
        )

        search_options = {
            "vector_queries": [vector_query],
            "top": top
        }

        if filter:
            search_options["filter"] = filter

        # Execute search
        results = client.search(
            search_text=None,  # No text search, only vector
            **search_options
        )

        # Format results with scores
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No Title")
            content = result.get("content", "")[:500]
            score = result.get("@search.score", 0)

            formatted_results.append(
                f"Result {i} (Score: {score:.3f}):\n"
                f"  Title: {title}\n"
                f"  Content: {content}...\n"
            )

        if not formatted_results:
            return f"No results found for vector query: '{query}'"

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error performing vector search: {str(e)}"


def create_hybrid_search_tool(config: Dict[str, Any], index_name: str = "documents"):
    """
    Create a hybrid search tool combining text and vector search.

    Args:
        config: Azure configuration dictionary
        index_name: Name of the search index

    Returns:
        A function for hybrid search
    """
    def hybrid_search(query: str, top: int = 5, alpha: float = 0.5) -> str:
        """
        Perform hybrid search combining text and vector search.

        Args:
            query: The search query
            top: Number of results to return
            alpha: Weight for text vs vector search (0=vector only, 1=text only)

        Returns:
            Combined and reranked search results
        """
        try:
            # Get both text and vector results
            text_results = azure_search_tool(config, query, top=top*2, index_name=index_name)
            vector_results = azure_vector_search_tool(config, query, top=top*2, index_name=index_name)

            # Simple combination - in production, you'd want more sophisticated reranking
            combined = f"=== Text Search Results (weight: {alpha}) ===\n{text_results}\n\n"
            combined += f"=== Vector Search Results (weight: {1-alpha}) ===\n{vector_results}"

            return combined

        except Exception as e:
            return f"Error in hybrid search: {str(e)}"

    # Set function name for LangChain compatibility
    hybrid_search.__name__ = "hybrid_search"

    return hybrid_search


# ============================================================================
# CLEAN WRAPPER FUNCTIONS (No config parameter needed!)
# ============================================================================

def search_documents(query: str, top: int = 5, search_fields: Optional[str] = None, index_name: str = "documents") -> str:
    """
    Search documents in Azure Cognitive Search.
    Uses the global configuration set via set_config().

    Args:
        query: The search query
        top: Number of results to return (default: 5)
        search_fields: Comma-separated list of fields to search
        index_name: Name of the search index (default: "documents")

    Returns:
        Formatted search results as a string
    """
    return azure_search_tool.invoke({"query": query, "top": top, "search_fields": search_fields, "index_name": index_name})


def vector_search_documents(query: str, top: int = 5, filter: Optional[str] = None, index_name: str = "documents") -> str:
    """
    Perform vector search using Azure Cognitive Search with embeddings.
    Uses the global configuration set via set_config().

    Args:
        query: The search query to vectorize
        top: Number of results to return (default: 5)
        filter: OData filter expression
        index_name: Name of the search index (default: "documents")

    Returns:
        Formatted search results with relevance scores
    """
    return azure_vector_search_tool.invoke({"query": query, "top": top, "filter": filter, "index_name": index_name})


def hybrid_search_documents(query: str, top: int = 5, alpha: float = 0.5, index_name: str = "documents") -> str:
    """
    Perform hybrid search combining text and vector search.
    Uses the global configuration set via set_config().

    Args:
        query: The search query
        top: Number of results to return
        alpha: Weight for text vs vector search (0=vector only, 1=text only)
        index_name: Name of the search index

    Returns:
        Combined and reranked search results
    """
    config = get_config()
    tool = create_hybrid_search_tool(config, index_name)
    return tool(query, top, alpha)