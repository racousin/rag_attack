"""Simple Tool Agent - Clean implementation with LangGraph"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Literal
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
import operator


class VerboseLevel(str, Enum):
    """Verbosity levels for agent output"""
    SILENT = "silent"    # No output
    MINIMAL = "minimal"  # Only final answer
    NORMAL = "normal"    # Tools used + final answer
    VERBOSE = "verbose"  # Full details


class AgentState(TypedDict):
    """State for the agent workflow"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Default prompts
DEFAULT_SYSTEM_PROMPT = """Tu es un assistant intelligent qui utilise des outils pour répondre aux questions.
Utilise les outils disponibles pour obtenir des informations précises et détaillées.
Réponds toujours dans la langue de la question."""


def create_llm(config: Dict[str, Any], temperature: float = 0.0) -> AzureChatOpenAI:
    """Create Azure OpenAI LLM instance"""
    return AzureChatOpenAI(
        azure_endpoint=config["openai_endpoint"],
        api_key=config["openai_key"],
        azure_deployment=config["chat_deployment"],
        api_version="2024-02-01",
        temperature=temperature
    )


def display_graph(graph, return_image: bool = False):
    """
    Display a LangGraph workflow as a visual diagram.

    Args:
        graph: A compiled LangGraph workflow
        return_image: If True, return the Image object instead of displaying

    Returns:
        Image object if return_image=True, otherwise displays in notebook
    """
    try:
        from IPython.display import Image, display as ipython_display

        if hasattr(graph, 'get_graph'):
            image_data = graph.get_graph().draw_mermaid_png()
        else:
            image_data = graph.draw_mermaid_png()

        img = Image(image_data)

        if return_image:
            return img
        else:
            ipython_display(img)

    except ImportError:
        print("IPython required. Install with: pip install ipython")
    except Exception as e:
        print(f"Could not display graph: {e}")


class SimpleAgent:
    """
    Simple tool-calling agent using LangGraph.

    The agent iteratively:
    1. Receives a question
    2. Decides which tool(s) to call
    3. Executes tools and observes results
    4. Repeats until it has enough information
    5. Generates a final answer

    Features:
    - Customizable system prompt
    - Adjustable verbosity levels
    - Graph visualization
    - Tool iteration limits
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        verbose: Literal["silent", "minimal", "normal", "verbose"] = "normal"
    ):
        """
        Initialize the SimpleAgent.

        Args:
            config: Azure configuration with OpenAI credentials
            tools: List of LangChain tools the agent can use
            system_prompt: Custom system prompt (uses default if None)
            max_iterations: Maximum tool calls before forcing answer (default: 5)
            verbose: Verbosity level - "silent", "minimal", "normal", "verbose"
        """
        self.config = config
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = VerboseLevel(verbose)

        # LLM setup
        self.llm = create_llm(config)
        self.llm_with_tools = self.llm.bind_tools(tools)

        # Prompts
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Tracking
        self._last_run = {}
        self._current_token_usage = {}

        # Build graph
        self.graph = self._build_graph()

    def _accumulate_tokens(self, response) -> None:
        """Accumulate token usage from an LLM response"""
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            usage = response.response_metadata['token_usage']
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    self._current_token_usage[key] = self._current_token_usage.get(key, 0) + value

    def get_last_run(self) -> Dict[str, Any]:
        """Get details from the last run including token usage"""
        return self._last_run.copy()

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt"""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set a new system prompt"""
        self._system_prompt = value

    def get_prompts(self) -> Dict[str, str]:
        """Get all configurable prompts"""
        return {
            "system_prompt": self._system_prompt
        }

    def set_prompts(self, system_prompt: Optional[str] = None):
        """
        Set custom prompts for the agent.

        Args:
            system_prompt: Main system instruction
        """
        if system_prompt is not None:
            self._system_prompt = system_prompt

    def _log(self, message: str, level: VerboseLevel = VerboseLevel.NORMAL):
        """Print message if verbosity level allows"""
        level_order = [VerboseLevel.SILENT, VerboseLevel.MINIMAL, VerboseLevel.NORMAL, VerboseLevel.VERBOSE]
        if level_order.index(self.verbose) >= level_order.index(level):
            print(message)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        # Edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent node - decides actions or generates answer"""
        messages = state["messages"]

        # Count tool calls
        tool_call_count = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls
        )

        self._log(f"\n  Iteration {tool_call_count + 1}/{self.max_iterations}", VerboseLevel.VERBOSE)

        # Check max iterations
        if tool_call_count >= self.max_iterations:
            self._log(f"  Max iterations reached. Generating answer...", VerboseLevel.VERBOSE)
            final_prompt = [
                {"role": "system", "content": "Provide a final answer based on gathered information. Do not use more tools."},
                *messages
            ]
            response = self.llm.invoke(final_prompt)
            self._accumulate_tokens(response)
            return {"messages": [response]}

        # Add system prompt for first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [{"role": "system", "content": self._system_prompt}, messages[0]]

        # Get LLM response
        response = self.llm_with_tools.invoke(messages)
        self._accumulate_tokens(response)

        # Log tool usage
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tc['name'] for tc in response.tool_calls]
            self._log(f"  Tools: {', '.join(tool_names)}", VerboseLevel.NORMAL)
        else:
            self._log(f"  Generating answer...", VerboseLevel.VERBOSE)

        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue or end"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def invoke(self, question: str) -> str:
        """
        Invoke the agent with a question.

        Args:
            question: The question to answer

        Returns:
            The agent's response string
        """
        self._current_token_usage = {}  # Reset token tracking

        # Header
        self._log(f"\n{'='*50}", VerboseLevel.NORMAL)
        self._log(f"SimpleAgent", VerboseLevel.NORMAL)
        self._log(f"{'='*50}", VerboseLevel.NORMAL)
        self._log(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}", VerboseLevel.NORMAL)
        self._log(f"Tools: {', '.join([t.name for t in self.tools])}", VerboseLevel.VERBOSE)
        self._log(f"Max iterations: {self.max_iterations}", VerboseLevel.VERBOSE)

        # Execute
        initial_state = {"messages": [HumanMessage(content=question)]}
        recursion_limit = max(self.max_iterations * 3 + 10, 30)

        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit}
        )

        # Extract answer
        response_content = "No response generated"
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and not message.tool_calls:
                response_content = message.content
                break

        self._last_run = {
            "question": question,
            "response": response_content,
            "token_usage": self._current_token_usage.copy()
        }

        self._log(f"{'='*50}\n", VerboseLevel.NORMAL)
        return response_content

    def stream(self, question: str):
        """Stream the agent's response chunk by chunk"""
        initial_state = {"messages": [HumanMessage(content=question)]}
        recursion_limit = max(self.max_iterations * 3 + 10, 30)

        for chunk in self.graph.stream(initial_state, config={"recursion_limit": recursion_limit}):
            yield chunk

    def display_graph(self, return_image: bool = False):
        """Display the agent's workflow as a visual diagram"""
        return display_graph(self.graph, return_image=return_image)

    def __repr__(self) -> str:
        return f"SimpleAgent(tools={[t.name for t in self.tools]}, max_iterations={self.max_iterations}, verbose={self.verbose.value})"
