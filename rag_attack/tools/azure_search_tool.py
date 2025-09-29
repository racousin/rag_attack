"""Azure Cognitive Search tool for RAG"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ..utils.config import get_search_client


class SearchQuery(BaseModel):
    """Input schema for Azure Search tool"""
    query: str = Field(description="The search query to execute")
    top: int = Field(default=5, description="Number of results to return")
    search_fields: Optional[str] = Field(default=None, description="Comma-separated list of fields to search")


@tool
def azure_search_tool(query: str, top: int = 5, search_fields: Optional[str] = None) -> str:
    """
    Search documents in Azure Cognitive Search.

    Args:
        query: The search query
        top: Number of results to return (default: 5)
        search_fields: Comma-separated list of fields to search

    Returns:
        Formatted search results as a string
    """
    try:
        client = get_search_client()

        # Prepare search options
        search_options = {
            "top": top,
            "include_total_count": True,
            "search_mode": "any"
        }

        if search_fields:
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
def azure_vector_search_tool(query: str, top: int = 5, filter: Optional[str] = None) -> str:
    """
    Perform vector search using Azure Cognitive Search with embeddings.

    Args:
        query: The search query to vectorize
        top: Number of results to return (default: 5)
        filter: OData filter expression

    Returns:
        Formatted search results with relevance scores
    """
    try:
        from openai import AzureOpenAI
        from ..utils.config import load_azure_config
        import numpy as np

        config = load_azure_config()

        # Get embeddings for query
        openai_client = AzureOpenAI(
            api_key=config["openai_key"],
            api_version="2024-02-01",
            azure_endpoint=config["openai_endpoint"]
        )

        # Generate embedding for the query
        response = openai_client.embeddings.create(
            input=query,
            model="textemb3small"  # Using the deployment name from config
        )

        query_embedding = response.data[0].embedding

        # Get search client
        client = get_search_client()

        # Prepare vector search
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top,
            fields="content_vector"  # Assuming this field exists
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


def create_hybrid_search_tool():
    """
    Create a hybrid search tool combining text and vector search.

    Returns:
        A LangChain tool for hybrid search
    """
    @tool
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
            text_results = azure_search_tool(query, top=top*2)
            vector_results = azure_vector_search_tool(query, top=top*2)

            # Simple combination - in production, you'd want more sophisticated reranking
            combined = f"=== Text Search Results (weight: {alpha}) ===\n{text_results}\n\n"
            combined += f"=== Vector Search Results (weight: {1-alpha}) ===\n{vector_results}"

            return combined

        except Exception as e:
            return f"Error in hybrid search: {str(e)}"

    return hybrid_search