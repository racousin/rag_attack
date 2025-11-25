"""Reflection Agent - Agent with critique and improvement loop"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Literal
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
import operator

from .simple_agent import create_llm, display_graph, VerboseLevel


class ReflectionState(TypedDict):
    """State for the reflection agent workflow"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    initial_response: Optional[str]
    critique: Optional[str]
    final_response: Optional[str]


# Default prompts
DEFAULT_SYSTEM_PROMPT = """Tu es un assistant intelligent qui utilise des outils pour répondre aux questions.
Utilise les outils disponibles pour obtenir des informations précises et détaillées.
Réponds toujours dans la langue de la question."""

DEFAULT_CRITIQUE_PROMPT = """Analyse cette réponse et identifie:
1. Les points forts
2. Les points à améliorer (clarté, complétude, précision)
3. Les informations manquantes

Question: {question}
Réponse: {response}

Critique:"""

DEFAULT_IMPROVE_PROMPT = """Améliore cette réponse en tenant compte de la critique.

Question: {question}
Réponse initiale: {response}
Critique: {critique}

Réponse améliorée:"""


class ReflectionAgent:
    """
    Agent with reflection loop that critiques and improves its responses.

    The workflow is:
    1. Generate initial response (using tools if needed)
    2. Critique the response
    3. Generate improved response

    This pattern produces higher quality answers for complex questions
    at the cost of additional LLM calls.

    Features:
    - Customizable prompts (system, critique, improve)
    - Adjustable verbosity levels
    - Graph visualization
    - Access to critique and improvement reasoning
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        critique_prompt: Optional[str] = None,
        improve_prompt: Optional[str] = None,
        max_iterations: int = 5,
        verbose: Literal["silent", "minimal", "normal", "verbose"] = "normal"
    ):
        """
        Initialize the ReflectionAgent.

        Args:
            config: Azure configuration with OpenAI credentials
            tools: List of LangChain tools the agent can use
            system_prompt: Custom system prompt
            critique_prompt: Custom critique prompt (use {question} and {response} placeholders)
            improve_prompt: Custom improvement prompt (use {question}, {response}, {critique} placeholders)
            max_iterations: Maximum tool calls for initial response (default: 5)
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
        self._critique_prompt = critique_prompt or DEFAULT_CRITIQUE_PROMPT
        self._improve_prompt = improve_prompt or DEFAULT_IMPROVE_PROMPT

        # Last run info (for inspection)
        self._last_initial_response = None
        self._last_critique = None
        self._last_improved_response = None

        # Build graph
        self.graph = self._build_graph()

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
            "system_prompt": self._system_prompt,
            "critique_prompt": self._critique_prompt,
            "improve_prompt": self._improve_prompt
        }

    def set_prompts(
        self,
        system_prompt: Optional[str] = None,
        critique_prompt: Optional[str] = None,
        improve_prompt: Optional[str] = None
    ):
        """
        Set custom prompts for the agent.

        Args:
            system_prompt: Main system instruction
            critique_prompt: Template for critique (use {question}, {response})
            improve_prompt: Template for improvement (use {question}, {response}, {critique})
        """
        if system_prompt is not None:
            self._system_prompt = system_prompt
        if critique_prompt is not None:
            self._critique_prompt = critique_prompt
        if improve_prompt is not None:
            self._improve_prompt = improve_prompt

    def get_last_run(self) -> Dict[str, Optional[str]]:
        """Get details from the last invocation"""
        return {
            "initial_response": self._last_initial_response,
            "critique": self._last_critique,
            "improved_response": self._last_improved_response
        }

    def _log(self, message: str, level: VerboseLevel = VerboseLevel.NORMAL):
        """Print message if verbosity level allows"""
        level_order = [VerboseLevel.SILENT, VerboseLevel.MINIMAL, VerboseLevel.NORMAL, VerboseLevel.VERBOSE]
        if level_order.index(self.verbose) >= level_order.index(level):
            print(message)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with reflection"""
        workflow = StateGraph(ReflectionState)

        # Nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("improve", self._improve_node)

        # Entry point
        workflow.set_entry_point("generate")

        # Edges for tool loop
        workflow.add_conditional_edges(
            "generate",
            self._should_use_tools,
            {"tools": "tools", "critique": "critique"}
        )
        workflow.add_edge("tools", "generate")

        # Reflection edges
        workflow.add_edge("critique", "improve")
        workflow.add_edge("improve", END)

        return workflow.compile()

    def _generate_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Generate initial response using tools"""
        messages = state["messages"]

        # Count tool calls
        tool_call_count = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls
        )

        self._log(f"  Iteration {tool_call_count + 1}/{self.max_iterations}", VerboseLevel.VERBOSE)

        # Check max iterations
        if tool_call_count >= self.max_iterations:
            self._log(f"  Max iterations reached. Generating response...", VerboseLevel.VERBOSE)
            final_prompt = [
                {"role": "system", "content": "Provide a response based on gathered information. Do not use more tools."},
                *messages
            ]
            response = self.llm.invoke(final_prompt)
            return {
                "messages": [response],
                "initial_response": response.content
            }

        # Add system prompt for first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [{"role": "system", "content": self._system_prompt}, messages[0]]

        # Get LLM response
        response = self.llm_with_tools.invoke(messages)

        # Log tool usage
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tc['name'] for tc in response.tool_calls]
            self._log(f"  Tools: {', '.join(tool_names)}", VerboseLevel.NORMAL)
            return {"messages": [response]}
        else:
            return {
                "messages": [response],
                "initial_response": response.content
            }

    def _should_use_tools(self, state: ReflectionState) -> str:
        """Decide whether to use tools or move to critique"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "critique"

    def _critique_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Critique the initial response"""
        self._log(f"\n  Critiquing response...", VerboseLevel.NORMAL)

        # Get question and response
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        initial_response = state.get("initial_response", "")

        # Generate critique
        critique_prompt = self._critique_prompt.format(
            question=question,
            response=initial_response
        )
        critique_response = self.llm.invoke(critique_prompt)
        critique = critique_response.content

        self._log(f"\n  Critique:", VerboseLevel.VERBOSE)
        self._log(f"  {critique[:200]}...", VerboseLevel.VERBOSE)

        return {"critique": critique}

    def _improve_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Generate improved response based on critique"""
        self._log(f"  Improving response...", VerboseLevel.NORMAL)

        # Get question
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        initial_response = state.get("initial_response", "")
        critique = state.get("critique", "")

        # Generate improved response
        improve_prompt = self._improve_prompt.format(
            question=question,
            response=initial_response,
            critique=critique
        )
        improved_response = self.llm.invoke(improve_prompt)

        return {"final_response": improved_response.content}

    def invoke(self, question: str) -> str:
        """
        Invoke the agent with a question.

        The agent will:
        1. Generate an initial response (using tools)
        2. Critique the response
        3. Return an improved response

        Args:
            question: The question to answer

        Returns:
            The improved response string
        """
        # Header
        self._log(f"\n{'='*50}", VerboseLevel.NORMAL)
        self._log(f"ReflectionAgent", VerboseLevel.NORMAL)
        self._log(f"{'='*50}", VerboseLevel.NORMAL)
        self._log(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}", VerboseLevel.NORMAL)
        self._log(f"Tools: {', '.join([t.name for t in self.tools])}", VerboseLevel.VERBOSE)

        self._log(f"\n  Phase 1: Initial Response", VerboseLevel.NORMAL)

        # Execute
        initial_state: ReflectionState = {
            "messages": [HumanMessage(content=question)],
            "initial_response": None,
            "critique": None,
            "final_response": None
        }
        recursion_limit = max(self.max_iterations * 3 + 20, 40)

        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit}
        )

        # Store for inspection
        self._last_initial_response = result.get("initial_response")
        self._last_critique = result.get("critique")
        self._last_improved_response = result.get("final_response")

        self._log(f"\n  Phase 2: Critique", VerboseLevel.NORMAL)
        self._log(f"\n  Phase 3: Improvement", VerboseLevel.NORMAL)
        self._log(f"{'='*50}\n", VerboseLevel.NORMAL)

        return result.get("final_response", "No response generated")

    def invoke_with_details(self, question: str) -> Dict[str, str]:
        """
        Invoke the agent and return all intermediate results.

        Args:
            question: The question to answer

        Returns:
            Dictionary with initial_response, critique, and improved_response
        """
        final_answer = self.invoke(question)
        return {
            "question": question,
            "initial_response": self._last_initial_response,
            "critique": self._last_critique,
            "improved_response": final_answer
        }

    def display_graph(self, return_image: bool = False):
        """Display the agent's workflow as a visual diagram"""
        return display_graph(self.graph, return_image=return_image)

    def __repr__(self) -> str:
        return f"ReflectionAgent(tools={[t.name for t in self.tools]}, max_iterations={self.max_iterations}, verbose={self.verbose.value})"
