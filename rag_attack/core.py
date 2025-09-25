"""Core components for RAG implementations"""

from typing import Dict, List, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import json


class AgentState:
    """Base state for agents"""

    def __init__(self):
        self.messages: List[BaseMessage] = []
        self.context: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_message(self, message: BaseMessage):
        """Add a message to the conversation"""
        self.messages.append(message)

    def get_last_message(self) -> Optional[BaseMessage]:
        """Get the last message in the conversation"""
        return self.messages[-1] if self.messages else None

    def update_context(self, key: str, value: Any):
        """Update context with new information"""
        self.context[key] = value


def create_basic_graph(agent_func, tools=None):
    """Create a basic StateGraph with a single agent node"""

    graph = StateGraph(Dict[str, Any])

    def agent_node(state):
        """Process state through agent"""
        result = agent_func(state, tools)
        return result

    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")

    def should_continue(state):
        """Decide whether to continue or end"""
        if "should_continue" in state:
            return "agent" if state["should_continue"] else END
        return END

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "agent",
            END: END
        }
    )

    return graph.compile()


def format_docs(docs: List[str]) -> str:
    """Format documents for context"""
    if not docs:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"Document {i}:\n{doc}\n")

    return "\n".join(formatted)


def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM response"""
    tool_calls = []

    # Simple parsing logic - can be enhanced
    if "<tool>" in response:
        # Extract tool calls from response
        import re
        pattern = r'<tool>(.*?)</tool>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

    return tool_calls