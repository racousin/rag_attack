"""RAG tool for searching VéloCorp documents in Azure Cognitive Search"""
from typing import Optional
from langchain_core.tools import tool
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from ..utils.config import get_config


@tool
def get_document_rag(
    query: str,
    search_type: str = "keyword",
    top: int = 5
) -> str:
    """Search VéloCorp documents using Azure Cognitive Search.

    The index contains: FAQs, user manuals, emails, call transcripts about VéloCorp bikes.

    Args:
        query: Search query (keywords or natural language question)
        search_type: Type of search to perform:
            - "keyword": BM25 keyword search (exact term matching)
            - "vector": Semantic vector search (meaning-based)
            - "hybrid": Combined keyword + vector search
        top: Number of results to return (default: 5)

    Returns:
        Formatted search results

    When to use each search_type:
        - "keyword": Exact terms like "E-City", "garantie 2 ans", product names
        - "vector": Conceptual queries like "how to maintain battery", "brake problems"
        - "hybrid": When unsure, combines both approaches

    Examples:
        get_document_rag("E-City specifications", search_type="keyword")
        get_document_rag("comment entretenir la batterie", search_type="vector")
        get_document_rag("problèmes de freins", search_type="hybrid")
    """
    config = get_config()
    search_type = search_type.lower().strip()

    try:
        index_name = config.get("search_index", "documents")
        client = SearchClient(
            endpoint=config["search_endpoint"],
            index_name=index_name,
            credential=AzureKeyCredential(config["search_key"])
        )

        if search_type == "keyword":
            return _keyword_search(client, query, top)
        elif search_type == "vector":
            return _vector_search(client, query, top, config)
        elif search_type == "hybrid":
            return _hybrid_search(client, query, top, config)
        else:
            return f"Error: Unknown search_type '{search_type}'. Must be: keyword, vector, or hybrid"

    except Exception as e:
        return f"Error searching documents: {str(e)}"


def _keyword_search(client: SearchClient, query: str, top: int) -> str:
    """BM25 keyword search"""
    results = client.search(
        search_text=query,
        top=top,
        include_total_count=True,
        search_mode="any"
    )

    return _format_results(results, "Keyword Search")


def _vector_search(client: SearchClient, query: str, top: int, config: dict) -> str:
    """Semantic vector search"""
    try:
        # Fix for PyTorch 2.9+ LRScheduler compatibility
        import torch.optim.lr_scheduler as lr_scheduler
        if not hasattr(lr_scheduler, 'LRScheduler'):
            lr_scheduler.LRScheduler = lr_scheduler._LRScheduler

        from langchain_huggingface import HuggingFaceEmbeddings
        from azure.search.documents.models import VectorizedQuery

        embedding_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        query_embedding = embedding_fn.embed_query(query)

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top,
            fields="embedding"
        )

        results = client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top
        )

        return _format_results(results, "Vector Search")

    except ImportError as e:
        return f"Error: Import failed: {str(e)}. Install with: pip install langchain-huggingface"
    except Exception as e:
        return f"Error in vector search: {str(e)}"


def _hybrid_search(client: SearchClient, query: str, top: int, config: dict) -> str:
    """Combined keyword + vector search"""
    try:
        # Fix for PyTorch 2.9+ LRScheduler compatibility
        import torch.optim.lr_scheduler as lr_scheduler
        if not hasattr(lr_scheduler, 'LRScheduler'):
            lr_scheduler.LRScheduler = lr_scheduler._LRScheduler

        from langchain_huggingface import HuggingFaceEmbeddings
        from azure.search.documents.models import VectorizedQuery

        embedding_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        query_embedding = embedding_fn.embed_query(query)

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top,
            fields="embedding"
        )

        results = client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top,
            search_mode="any"
        )

        return _format_results(results, "Hybrid Search")

    except ImportError as e:
        return f"Error: Import failed: {str(e)}. Install with: pip install langchain-huggingface"
    except Exception as e:
        return f"Error in hybrid search: {str(e)}"


def _format_results(results, search_type: str) -> str:
    """Format search results"""
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "No Title")
        content = result.get("content", "")[:500]
        category = result.get("category", "Unknown")
        score = result.get("@search.score", 0)

        formatted.append(
            f"Result {i} (score: {score:.2f}):\n"
            f"  Title: {title}\n"
            f"  Category: {category}\n"
            f"  Content: {content}...\n"
        )

    if not formatted:
        return f"No results found ({search_type})"

    return f"=== {search_type} Results ===\n\n" + "\n".join(formatted)
