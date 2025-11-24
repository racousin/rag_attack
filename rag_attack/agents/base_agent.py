"""Base agent implementation using LangGraph"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
import operator


class AgentState(TypedDict):
    """State for the agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def create_llm(config: Dict[str, Any], temperature: float = 0.0) -> AzureChatOpenAI:
    """Create Azure OpenAI LLM instance

    Args:
        config: Azure configuration dictionary with OpenAI credentials
        temperature: Temperature for the LLM

    Returns:
        AzureChatOpenAI instance
    """
    return AzureChatOpenAI(
        azure_endpoint=config["openai_endpoint"],
        api_key=config["openai_key"],
        azure_deployment=config["chat_deployment"],
        api_version="2024-02-01",
        temperature=temperature
    )


class SimpleToolAgent:
    """Simple agent that can use tools to answer questions"""

    def __init__(self, config: Dict[str, Any], tools: List[BaseTool], system_prompt: str = None, max_iterations: int = 15, verbose: bool = False):
        """
        Initialize the agent with tools.

        Args:
            config: Azure configuration dictionary
            tools: List of LangChain tools the agent can use
            system_prompt: Optional system prompt for the agent
            max_iterations: Maximum number of tool calls before stopping (default: 15)
            verbose: Whether to print progress information (default: False)
        """
        self.config = config
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.llm = create_llm(config)

        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(tools)

        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that can use various tools to answer questions. "
                "Use the tools when needed to provide accurate and detailed answers."
            )
        self.system_prompt = system_prompt

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent node that decides what to do"""
        messages = state["messages"]

        # Count tool calls to enforce max_iterations
        tool_call_count = sum(1 for msg in messages if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls)

        if self.verbose:
            print(f"\n🤖 Agent thinking... (Tool calls so far: {tool_call_count}/{self.max_iterations})")

        # Check if we've exceeded max iterations
        if tool_call_count >= self.max_iterations:
            if self.verbose:
                print(f"⚠️ Reached max iterations ({self.max_iterations}). Generating final answer with available information.")
            # Force a final answer without tool calls
            final_prompt = [
                {"role": "system", "content": "Provide a final answer based on the information gathered so far. Do not use any more tools."},
                *messages
            ]
            response = self.llm.invoke(final_prompt)
            return {"messages": [response]}

        # Add system prompt if this is the first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [
                {"role": "system", "content": self.system_prompt},
                messages[0]
            ]

        # Get response from LLM
        response = self.llm_with_tools.invoke(messages)

        if self.verbose:
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 Agent wants to use tools: {[tc['name'] for tc in response.tool_calls]}")
            else:
                print(f"✅ Agent provided final answer")

        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end"""
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        # Otherwise, end
        return "end"

    def invoke(self, question: str) -> str:
        """
        Invoke the agent with a question.

        Args:
            question: The question to ask

        Returns:
            The agent's response
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚀 SimpleToolAgent Starting")
            print(f"{'='*60}")
            print(f"❓ Question: {question}")
            print(f"🔧 Available tools: {', '.join([t.name for t in self.tools])}")
            print(f"🔄 Max iterations: {self.max_iterations}")
            print(f"{'='*60}")

        initial_state = {
            "messages": [HumanMessage(content=question)]
        }

        # Set recursion limit based on max_iterations
        # Each iteration goes through agent -> tools -> agent (2 nodes)
        recursion_limit = max(self.max_iterations * 3 + 10, 30)

        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit}
        )

        # Extract the final answer
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and not message.tool_calls:
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"✨ Final Answer Generated")
                    print(f"{'='*60}\n")
                return message.content

        return "No response generated"

    def stream(self, question: str):
        """
        Stream the agent's response.

        Args:
            question: The question to ask

        Yields:
            Chunks of the response
        """
        initial_state = {
            "messages": [HumanMessage(content=question)]
        }

        # Set recursion limit based on max_iterations
        recursion_limit = max(self.max_iterations * 3 + 10, 30)

        for chunk in self.graph.stream(initial_state, config={"recursion_limit": recursion_limit}):
            yield chunk