"""Mock agent for testing and demo purposes without Azure"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import FakeListLLM
from langchain_core.tools import BaseTool
import operator


class MockAgentState(TypedDict):
    """State for the mock agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


class MockAgent:
    """Mock agent that works without Azure OpenAI for testing"""

    def __init__(self, tools: List[BaseTool], system_prompt: str = None):
        """
        Initialize mock agent with tools.

        Args:
            tools: List of tools the agent can use
            system_prompt: Optional system prompt for the agent
        """
        self.tools = tools

        # Use a mock LLM for testing
        self.llm = FakeListLLM(
            responses=[
                "I'll search for that information.",
                "Based on the search results, here's what I found:",
                "Let me query the database.",
                "According to the data:",
                "Here's a summary of the findings:"
            ]
        )

        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        self.system_prompt = system_prompt

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the workflow graph"""
        workflow = StateGraph(MockAgentState)

        # Add nodes
        workflow.add_node("process", self._process_node)

        # Set entry and exit
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)

        return workflow.compile()

    def _process_node(self, state: MockAgentState) -> Dict[str, Any]:
        """Process the request with mock responses"""
        messages = state["messages"]

        # Get the user's question
        user_question = messages[-1].content if messages else ""

        # Simulate tool usage if tools are available
        tool_results = []
        if self.tools:
            for tool in self.tools[:1]:  # Use first tool for demo
                try:
                    result = tool.invoke(user_question)
                    tool_results.append(f"Tool {tool.name}: {result}")
                except:
                    tool_results.append(f"Tool {tool.name}: Mock result for '{user_question}'")

        # Create a mock response
        if tool_results:
            response = f"Based on the tools, here's what I found:\n" + "\n".join(tool_results)
        else:
            response = f"Mock response for: {user_question}"

        return {"messages": [AIMessage(content=response)]}

    def invoke(self, question: str) -> str:
        """
        Invoke the mock agent with a question.

        Args:
            question: The question to ask

        Returns:
            The mock agent's response
        """
        initial_state = {
            "messages": [HumanMessage(content=question)]
        }

        result = self.graph.invoke(initial_state)

        # Extract the final answer
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                return message.content

        return "No response generated"


def create_demo_tools():
    """Create demo tools that work without Azure"""
    from langchain_core.tools import tool

    @tool
    def mock_search(query: str) -> str:
        """Mock search tool for demo"""
        mock_data = {
            "vélo": "VéloCorp propose plusieurs modèles de vélos électriques: Urban (1500€), Sport (2500€), et Elite (4500€).",
            "garantie": "Tous les vélos VéloCorp bénéficient d'une garantie de 2 ans pièces et main d'œuvre.",
            "batterie": "Les batteries lithium-ion offrent une autonomie de 50 à 120 km selon le modèle.",
            "livraison": "Livraison gratuite en France métropolitaine sous 5 jours ouvrés.",
            "entretien": "Service d'entretien annuel inclus la première année.",
            "performance": "Moteur 250W conforme à la réglementation européenne, assistance jusqu'à 25 km/h."
        }

        query_lower = query.lower()
        results = []
        for key, value in mock_data.items():
            if key in query_lower:
                results.append(value)

        if results:
            return " ".join(results)
        return f"Aucune information trouvée pour: {query}"

    @tool
    def mock_sql_query(query: str) -> str:
        """Mock SQL query tool for demo"""
        if "orders" in query.lower():
            return """
            Recent Orders:
            OrderID | CustomerID | Product      | Amount  | Date
            --------|------------|--------------|---------|------------
            1001    | C456       | Urban        | 1500€   | 2024-01-15
            1002    | C789       | Sport        | 2500€   | 2024-01-16
            1003    | C123       | Elite        | 4500€   | 2024-01-17
            """
        elif "customers" in query.lower():
            return """
            Customer Statistics:
            - Total customers: 1,247
            - New this month: 89
            - Premium members: 312
            """
        return "Query executed successfully (mock data)"

    @tool
    def mock_api_call(endpoint: str) -> str:
        """Mock API call for demo"""
        endpoints = {
            "/health": "API Status: Healthy",
            "/crm/opportunities": "5 active opportunities worth 125,000€ total",
            "/weather": "Paris: 18°C, Partly cloudy"
        }
        return endpoints.get(endpoint, f"Mock response from {endpoint}")

    return [mock_search, mock_sql_query, mock_api_call]