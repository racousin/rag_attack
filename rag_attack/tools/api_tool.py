"""API tools for interacting with Azure Functions and external APIs"""
import json
import requests
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import config management from azure_search_tool
from .azure_search_tool import get_config


class APIRequest(BaseModel):
    """Input schema for API request tool"""
    endpoint: str = Field(description="The API endpoint path (e.g., '/crm/opportunities')")
    method: str = Field(default="GET", description="HTTP method (GET, POST, PUT, DELETE)")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Request body data")


def azure_function_api_tool(
    config: Dict[str, Any],
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call Azure Function API endpoints.

    Args:
        config: Azure configuration dictionary with 'api_base_url'
        endpoint: The API endpoint path (e.g., '/crm/opportunities')
        method: HTTP method (GET, POST, PUT, DELETE)
        params: Query parameters
        data: Request body data

    Returns:
        API response as formatted JSON string
    """
    try:
        base_url = config["api_base_url"]

        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # Construct full URL
        full_url = base_url.rstrip("/api") + "/api" + endpoint

        # Make request
        response = requests.request(
            method=method.upper(),
            url=full_url,
            params=params,
            json=data if data else None,
            timeout=30
        )

        # Check response
        response.raise_for_status()

        # Format response
        try:
            response_data = response.json()
            return json.dumps(response_data, indent=2)
        except json.JSONDecodeError:
            return response.text

    except requests.exceptions.RequestException as e:
        return f"API request error: {str(e)}"
    except Exception as e:
        return f"Error calling API: {str(e)}"


def search_api_tool(config: Dict[str, Any], query: str, top: int = 5) -> str:
    """
    Search using the Azure Function search API.

    Args:
        config: Azure configuration dictionary
        query: The search query
        top: Number of results to return

    Returns:
        Search results from the API
    """
    return azure_function_api_tool(
        config,
        endpoint="/search",
        params={"q": query, "top": top}
    )


def crm_opportunities_tool(
    config: Dict[str, Any],
    status: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> str:
    """
    Get CRM opportunities from the Azure Function API.

    Args:
        config: Azure configuration dictionary
        status: Filter by opportunity status
        min_value: Minimum opportunity value
        max_value: Maximum opportunity value

    Returns:
        CRM opportunities data
    """
    params = {}
    if status:
        params["status"] = status
    if min_value is not None:
        params["min_value"] = min_value
    if max_value is not None:
        params["max_value"] = max_value

    return azure_function_api_tool(
        config,
        endpoint="/crm/opportunities",
        params=params if params else None
    )


def weather_api_tool(city: str) -> str:
    """
    Get weather information for a city using OpenWeatherMap API.

    Args:
        city: Name of the city

    Returns:
        Weather information
    """
    try:
        # Using a free weather API (you might want to add API key to config)
        url = f"https://wttr.in/{city}?format=j1"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract key information
        current = data.get("current_condition", [{}])[0]
        weather_desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
        temp_c = current.get("temp_C", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_speed = current.get("windspeedKmph", "N/A")

        return (
            f"Weather in {city}:\n"
            f"  Description: {weather_desc}\n"
            f"  Temperature: {temp_c}°C\n"
            f"  Humidity: {humidity}%\n"
            f"  Wind Speed: {wind_speed} km/h"
        )

    except Exception as e:
        return f"Error getting weather data: {str(e)}"


def web_search_tool(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo (no API key required).

    Args:
        query: The search query
        num_results: Number of results to return

    Returns:
        Web search results
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

            if not results:
                return f"No results found for: {query}"

            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"Result {i}:\n"
                    f"  Title: {result.get('title', 'N/A')}\n"
                    f"  URL: {result.get('href', 'N/A')}\n"
                    f"  Snippet: {result.get('body', 'N/A')[:200]}...\n"
                )

            return "\n".join(formatted_results)

    except ImportError:
        return "DuckDuckGo search not available. Install with: pip install duckduckgo-search"
    except Exception as e:
        return f"Error searching the web: {str(e)}"


def create_api_tools(config: Dict[str, Any]):
    """
    Create a collection of API tools with config bound.

    Args:
        config: Azure configuration dictionary

    Returns:
        List of API-related tools with config bound
    """
    from functools import partial

    # Create partial functions and set their __name__ attribute for LangChain compatibility
    azure_api_partial = partial(azure_function_api_tool, config)
    azure_api_partial.__name__ = "azure_function_api_tool"

    search_api_partial = partial(search_api_tool, config)
    search_api_partial.__name__ = "search_api_tool"

    crm_partial = partial(crm_opportunities_tool, config)
    crm_partial.__name__ = "crm_opportunities_tool"

    return [
        azure_api_partial,
        search_api_partial,
        crm_partial,
        weather_api_tool,
        web_search_tool
    ]


# ============================================================================
# CLEAN WRAPPER FUNCTIONS (No config parameter needed!)
# ============================================================================

def call_azure_api(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call Azure Function API endpoints.
    Uses the global configuration set via set_config().

    Args:
        endpoint: The API endpoint path (e.g., '/crm/opportunities')
        method: HTTP method (GET, POST, PUT, DELETE)
        params: Query parameters
        data: Request body data

    Returns:
        API response as formatted JSON string
    """
    config = get_config()
    return azure_function_api_tool(config, endpoint, method, params, data)


def search_via_api(query: str, top: int = 5) -> str:
    """
    Search using the Azure Function search API.
    Uses the global configuration set via set_config().

    Args:
        query: The search query
        top: Number of results to return

    Returns:
        Search results from the API
    """
    config = get_config()
    return search_api_tool(config, query, top)


def get_crm_opportunities(
    status: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> str:
    """
    Get CRM opportunities from the Azure Function API.
    Uses the global configuration set via set_config().

    Args:
        status: Filter by opportunity status
        min_value: Minimum opportunity value
        max_value: Maximum opportunity value

    Returns:
        CRM opportunities data
    """
    config = get_config()
    return crm_opportunities_tool(config, status, min_value, max_value)


def get_weather(city: str) -> str:
    """
    Get weather information for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information
    """
    return weather_api_tool(city)


def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query
        num_results: Number of results to return

    Returns:
        Web search results
    """
    return web_search_tool(query, num_results)