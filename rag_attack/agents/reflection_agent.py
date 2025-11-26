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
    current_response: Optional[str]  # Current version being refined
    critique: Optional[str]
    final_response: Optional[str]
    reflection_count: int  # Track reflection loop iterations
    critiques_history: List[str]  # Store all critiques for inspection
    _evaluation: Optional[str]  # Internal: last evaluation result


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

DEFAULT_EVALUATE_PROMPT = """Évalue si cette réponse est suffisamment bonne pour être finale.

Question: {question}
Réponse: {response}
Critique précédente: {critique}

Réponds UNIQUEMENT par:
- "APPROVED" si la réponse est complète, précise et bien structurée
- "NEEDS_IMPROVEMENT" suivi d'une brève explication si des améliorations significatives sont encore possibles

Évaluation:"""


class ReflectionAgent:
    """
    Agent with iterative reflection loop that critiques and improves its responses.

    The workflow is:
    1. Generate initial response (using tools if needed)
    2. Critique the response
    3. Generate improved response
    4. Evaluate if response is good enough
       - If APPROVED → return final response
       - If NEEDS_IMPROVEMENT → loop back to step 2 (up to max_reflections)

    This critic loop pattern produces higher quality answers by iteratively
    refining responses until they meet quality standards.

    Features:
    - Iterative critique-improve loop with configurable max iterations
    - Customizable prompts (system, critique, improve, evaluate)
    - Adjustable verbosity levels
    - Graph visualization
    - Access to all critiques history and intermediate responses
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        critique_prompt: Optional[str] = None,
        improve_prompt: Optional[str] = None,
        evaluate_prompt: Optional[str] = None,
        max_iterations: int = 5,
        max_reflections: int = 3,
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
            evaluate_prompt: Custom evaluation prompt (use {question}, {response}, {critique} placeholders)
            max_iterations: Maximum tool calls for initial response (default: 5)
            max_reflections: Maximum critique-improve loops (default: 3)
            verbose: Verbosity level - "silent", "minimal", "normal", "verbose"
        """
        self.config = config
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_reflections = max_reflections
        self.verbose = VerboseLevel(verbose)

        # LLM setup
        self.llm = create_llm(config)
        self.llm_with_tools = self.llm.bind_tools(tools)

        # Prompts
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._critique_prompt = critique_prompt or DEFAULT_CRITIQUE_PROMPT
        self._improve_prompt = improve_prompt or DEFAULT_IMPROVE_PROMPT
        self._evaluate_prompt = evaluate_prompt or DEFAULT_EVALUATE_PROMPT

        # Last run info (for inspection)
        self._last_initial_response = None
        self._last_critique = None
        self._last_improved_response = None
        self._last_critiques_history = []

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
            "improve_prompt": self._improve_prompt,
            "evaluate_prompt": self._evaluate_prompt
        }

    def set_prompts(
        self,
        system_prompt: Optional[str] = None,
        critique_prompt: Optional[str] = None,
        improve_prompt: Optional[str] = None,
        evaluate_prompt: Optional[str] = None
    ):
        """
        Set custom prompts for the agent.

        Args:
            system_prompt: Main system instruction
            critique_prompt: Template for critique (use {question}, {response})
            improve_prompt: Template for improvement (use {question}, {response}, {critique})
            evaluate_prompt: Template for evaluation (use {question}, {response}, {critique})
        """
        if system_prompt is not None:
            self._system_prompt = system_prompt
        if critique_prompt is not None:
            self._critique_prompt = critique_prompt
        if improve_prompt is not None:
            self._improve_prompt = improve_prompt
        if evaluate_prompt is not None:
            self._evaluate_prompt = evaluate_prompt

    def get_last_run(self) -> Dict[str, Any]:
        """Get details from the last invocation"""
        return {
            "initial_response": self._last_initial_response,
            "final_critique": self._last_critique,
            "improved_response": self._last_improved_response,
            "critiques_history": self._last_critiques_history,
            "reflection_iterations": len(self._last_critiques_history)
        }

    def _log(self, message: str, level: VerboseLevel = VerboseLevel.NORMAL):
        """Print message if verbosity level allows"""
        level_order = [VerboseLevel.SILENT, VerboseLevel.MINIMAL, VerboseLevel.NORMAL, VerboseLevel.VERBOSE]
        if level_order.index(self.verbose) >= level_order.index(level):
            print(message)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with iterative reflection loop"""
        workflow = StateGraph(ReflectionState)

        # Nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("improve", self._improve_node)
        workflow.add_node("tools_reflect", ToolNode(self.tools))  # Tools available during reflection
        workflow.add_node("evaluate", self._evaluate_node)

        # Entry point
        workflow.set_entry_point("generate")

        # Edges for tool loop (generate phase)
        workflow.add_conditional_edges(
            "generate",
            self._should_use_tools,
            {"tools": "tools", "critique": "critique"}
        )
        workflow.add_edge("tools", "generate")

        # Reflection loop: critique → improve (with tool access) → evaluate → (critique or END)
        workflow.add_edge("critique", "improve")
        workflow.add_conditional_edges(
            "improve",
            self._should_use_tools_or_evaluate,
            {"tools_reflect": "tools_reflect", "evaluate": "evaluate", "end": END}
        )
        workflow.add_edge("tools_reflect", "improve")  # Tool results go back to improve
        workflow.add_conditional_edges(
            "evaluate",
            self._evaluate_decision,
            {"critique": "critique", "end": END}
        )

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
                "initial_response": response.content,
                "current_response": response.content,
                "reflection_count": 0,
                "critiques_history": []
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
                "initial_response": response.content,
                "current_response": response.content,
                "reflection_count": 0,
                "critiques_history": []
            }

    def _should_use_tools(self, state: ReflectionState) -> str:
        """Decide whether to use tools or move to critique"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "critique"

    def _critique_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Critique the current response"""
        reflection_count = state.get("reflection_count", 0)
        self._log(f"\n  Critiquing response (iteration {reflection_count + 1})...", VerboseLevel.NORMAL)

        # Get question and current response
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        current_response = state.get("current_response", state.get("initial_response", ""))

        # Generate critique
        critique_prompt = self._critique_prompt.format(
            question=question,
            response=current_response
        )
        critique_response = self.llm.invoke(critique_prompt)
        critique = critique_response.content

        self._log(f"\n  Critique:", VerboseLevel.VERBOSE)
        self._log(f"  {critique[:200]}...", VerboseLevel.VERBOSE)

        # Add to history
        critiques_history = state.get("critiques_history", []) + [critique]

        return {
            "critique": critique,
            "critiques_history": critiques_history
        }

    def _improve_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Generate improved response based on critique - can use tools if needed"""
        reflection_count = state.get("reflection_count", 0)
        messages = state["messages"]

        # Check if last message is a tool result (continuing tool loop)
        last_msg = messages[-1] if messages else None
        is_tool_result = hasattr(last_msg, 'type') and last_msg.type == 'tool'

        if not is_tool_result:
            self._log(f"  Improving response (iteration {reflection_count + 1})...", VerboseLevel.NORMAL)

        # Get question
        question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        current_response = state.get("current_response", state.get("initial_response", ""))
        critique = state.get("critique", "")

        # Build improve prompt with tool access
        improve_prompt = self._improve_prompt.format(
            question=question,
            response=current_response,
            critique=critique
        )

        # Use LLM with tools - agent can fetch more info if critique identified gaps
        improve_messages = [
            {"role": "system", "content": f"{self._system_prompt}\n\nTu peux utiliser les outils si tu as besoin d'informations supplémentaires pour améliorer la réponse."},
            {"role": "user", "content": improve_prompt}
        ]

        # Add any tool results from previous iterations in this improvement phase
        for msg in messages:
            if hasattr(msg, 'tool_calls') or (hasattr(msg, 'type') and msg.type == 'tool'):
                improve_messages.append(msg)

        response = self.llm_with_tools.invoke(improve_messages)

        # If tool calls, return for tool execution
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tc['name'] for tc in response.tool_calls]
            self._log(f"    Fetching more info: {', '.join(tool_names)}", VerboseLevel.NORMAL)
            return {"messages": [response]}

        # No tool calls - we have the improved response
        return {
            "current_response": response.content,
            "reflection_count": reflection_count + 1
        }

    def _should_use_tools_or_evaluate(self, state: ReflectionState) -> str:
        """Decide whether to use tools, evaluate, or end"""
        messages = state["messages"]
        reflection_count = state.get("reflection_count", 0)

        # Check if last message has tool calls (improve wants to fetch more info)
        last_msg = messages[-1] if messages else None
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools_reflect"

        # If we've hit max reflections, end immediately
        if reflection_count >= self.max_reflections:
            self._log(f"  Max reflections ({self.max_reflections}) reached.", VerboseLevel.NORMAL)
            return "end"

        return "evaluate"

    def _evaluate_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Evaluate if the improved response is good enough"""
        reflection_count = state.get("reflection_count", 0)
        self._log(f"\n  Evaluating response quality...", VerboseLevel.NORMAL)

        # Get question and current response
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        current_response = state.get("current_response", "")
        critique = state.get("critique", "")

        # Generate evaluation
        evaluate_prompt = self._evaluate_prompt.format(
            question=question,
            response=current_response,
            critique=critique
        )
        evaluation = self.llm.invoke(evaluate_prompt).content

        self._log(f"  Evaluation: {evaluation[:100]}...", VerboseLevel.VERBOSE)

        # Store evaluation result in state for routing decision
        return {"_evaluation": evaluation}

    def _evaluate_decision(self, state: ReflectionState) -> str:
        """Route based on evaluation result"""
        evaluation = state.get("_evaluation", "")
        reflection_count = state.get("reflection_count", 0)

        # Check if approved or max reflections reached
        if "APPROVED" in evaluation.upper() or reflection_count >= self.max_reflections:
            if reflection_count >= self.max_reflections:
                self._log(f"  Max reflections reached, finalizing.", VerboseLevel.NORMAL)
            else:
                self._log(f"  Response approved after {reflection_count} iteration(s).", VerboseLevel.NORMAL)
            return "end"

        self._log(f"  Needs improvement, continuing reflection loop...", VerboseLevel.NORMAL)
        return "critique"

    def invoke(self, question: str) -> str:
        """
        Invoke the agent with a question.

        The agent will:
        1. Generate an initial response (using tools)
        2. Enter the reflection loop:
           - Critique the response
           - Improve based on critique
           - Evaluate if good enough
           - Loop back or finalize

        Args:
            question: The question to answer

        Returns:
            The final improved response string
        """
        # Header
        self._log(f"\n{'='*50}", VerboseLevel.NORMAL)
        self._log(f"ReflectionAgent (max {self.max_reflections} reflection loops)", VerboseLevel.NORMAL)
        self._log(f"{'='*50}", VerboseLevel.NORMAL)
        self._log(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}", VerboseLevel.NORMAL)
        self._log(f"Tools: {', '.join([t.name for t in self.tools])}", VerboseLevel.VERBOSE)

        self._log(f"\n  Phase 1: Generating initial response...", VerboseLevel.NORMAL)

        # Execute
        initial_state: ReflectionState = {
            "messages": [HumanMessage(content=question)],
            "initial_response": None,
            "current_response": None,
            "critique": None,
            "final_response": None,
            "reflection_count": 0,
            "critiques_history": [],
            "_evaluation": None
        }
        # Increased recursion limit to account for reflection loops
        recursion_limit = max(self.max_iterations * 3 + self.max_reflections * 6 + 20, 50)

        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit}
        )

        # Store for inspection
        self._last_initial_response = result.get("initial_response")
        self._last_critique = result.get("critique")
        self._last_improved_response = result.get("current_response")
        self._last_critiques_history = result.get("critiques_history", [])

        reflection_count = result.get("reflection_count", 0)
        self._log(f"\n  Completed with {reflection_count} reflection iteration(s)", VerboseLevel.NORMAL)
        self._log(f"{'='*50}\n", VerboseLevel.NORMAL)

        return result.get("current_response", result.get("initial_response", "No response generated"))

    def invoke_with_details(self, question: str) -> Dict[str, Any]:
        """
        Invoke the agent and return all intermediate results.

        Args:
            question: The question to answer

        Returns:
            Dictionary with initial_response, all critiques, and final response
        """
        final_answer = self.invoke(question)
        return {
            "question": question,
            "initial_response": self._last_initial_response,
            "final_response": final_answer,
            "final_critique": self._last_critique,
            "critiques_history": self._last_critiques_history,
            "reflection_iterations": len(self._last_critiques_history)
        }

    def display_graph(self, return_image: bool = False):
        """Display the agent's workflow as a visual diagram"""
        return display_graph(self.graph, return_image=return_image)

    def __repr__(self) -> str:
        return f"ReflectionAgent(tools={[t.name for t in self.tools]}, max_iterations={self.max_iterations}, max_reflections={self.max_reflections}, verbose={self.verbose.value})"
