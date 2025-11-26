"""Internet search tool using DuckDuckGo"""
from langchain_core.tools import tool


@tool
def get_internet_search(query: str, num_results: int = 5) -> str:
    """Search the internet using DuckDuckGo (no API key required).

    Use this tool for external information not available in VéloCorp's internal systems:
    - Market research and competitor analysis
    - Industry trends and news
    - General knowledge questions
    - External product comparisons

    Args:
        query: Search query
        num_results: Number of results to return (default: 5)

    Returns:
        Web search results with titles, URLs, and snippets

    Examples:
        get_internet_search("vélo électrique tendances 2024")
        get_internet_search("comparatif vélos cargo marché français")
        get_internet_search("réglementation vélos électriques France")
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

            if not results:
                return f"No results found for: {query}"

            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"Result {i}:\n"
                    f"  Title: {result.get('title', 'N/A')}\n"
                    f"  URL: {result.get('href', 'N/A')}\n"
                    f"  Snippet: {result.get('body', 'N/A')[:300]}...\n"
                )

            return "\n".join(formatted)

    except ImportError:
        return "Error: DuckDuckGo search not available. Install with: pip install ddgs"
    except Exception as e:
        return f"Error searching the web: {str(e)}"
