"""CRM tool for accessing VéloCorp CRM data via API"""
import json
import requests
from typing import Optional
from langchain_core.tools import tool

from ..utils.config import get_config


def _call_api(endpoint: str, params: Optional[dict] = None) -> str:
    """Internal helper to call the CRM API"""
    config = get_config()
    try:
        base_url = config.get("crm_base_url") or config.get("api_base_url", "")
        if not base_url:
            return "Error: CRM API URL not configured. Set 'crm_base_url' in config."

        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        full_url = base_url.rstrip("/api") + "/api" + endpoint

        headers = {}
        api_key = config.get("crm_api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.request(
            method="GET",
            url=full_url,
            params=params,
            headers=headers if headers else None,
            timeout=30
        )
        response.raise_for_status()

        try:
            return json.dumps(response.json(), indent=2)
        except json.JSONDecodeError:
            return response.text
    except requests.exceptions.RequestException as e:
        return f"API request error: {str(e)}"
    except Exception as e:
        return f"Error calling API: {str(e)}"


@tool
def get_crm(
    query_type: str,
    status: Optional[str] = None,
    min_value: Optional[float] = None,
    min_score: Optional[int] = None,
    region: Optional[str] = None,
    limit: int = 50
) -> str:
    """Get CRM data from VéloCorp CRM system.

    Args:
        query_type: Type of CRM data to retrieve. Must be one of:
            - "opportunities": Sales opportunities with pipeline metrics
            - "prospects": Leads and prospects with conversion metrics
            - "sales_reps": Sales representatives and their performance
            - "analytics": Global CRM analytics and KPIs
        status: Filter by status (for opportunities: "Proposition", "Négociation", "Gagné", "Perdu")
                (for prospects: "Nouveau", "Qualifié", "Contacté")
        min_value: Minimum value in euros (for opportunities)
        min_score: Minimum lead score 0-100 (for prospects)
        region: Filter by region (for sales_reps)
        limit: Maximum results to return (default: 50)

    Returns:
        CRM data as JSON string

    Examples:
        get_crm("analytics") - Get global CRM KPIs
        get_crm("opportunities", status="Proposition", min_value=1000) - Active opportunities > 1000€
        get_crm("prospects", min_score=70) - High-quality leads
        get_crm("sales_reps", region="Bretagne") - Sales reps in Bretagne
    """
    query_type = query_type.lower().strip()

    if query_type == "opportunities":
        params = {}
        if status:
            params["status"] = status
        if min_value is not None:
            params["min_value"] = min_value
        return _call_api("/crm/opportunites", params if params else None)

    elif query_type == "prospects":
        params = {"limit": limit}
        if status:
            params["status"] = status
        if min_score is not None:
            params["min_score"] = min_score
        return _call_api("/crm/prospects", params)

    elif query_type == "sales_reps":
        params = {}
        if region:
            params["region"] = region
        return _call_api("/crm/commerciaux", params if params else None)

    elif query_type == "analytics":
        return _call_api("/crm/analytics")

    else:
        return f"Error: Unknown query_type '{query_type}'. Must be one of: opportunities, prospects, sales_reps, analytics"
