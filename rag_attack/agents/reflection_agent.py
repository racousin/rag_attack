"""Reflection Agent - Generate → Critique → Loop or END"""
from typing import TypedDict, Annotated, Sequence, Dict, Any, Optional, Literal, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
import operator

from .simple_agent import create_llm, display_graph, VerboseLevel


# Default prompts
DEFAULT_SYSTEM_PROMPT = """Tu es un assistant intelligent qui utilise des outils pour répondre aux questions.
Utilise les outils disponibles pour obtenir des informations précises.
Réponds toujours dans la langue de la question."""

DEFAULT_CRITIQUE_PROMPT = """Évalue cette réponse à la question posée.

Question: {question}
Réponse: {response}

Réponds UNIQUEMENT avec un JSON:
{{"is_good": true/false, "feedback": "explication courte"}}

- is_good=true si la réponse est complète, précise et répond bien à la question
- is_good=false si des informations manquent ou si la réponse peut être améliorée"""


class ReflectionState(TypedDict):
    """State for the reflection agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    current_response: Optional[str]
    critique: Optional[str]
    iteration: int


class ReflectionAgent:
    """
    Agent with reflection: generate (with tools) → critique → loop or END

    Simple flow:
    1. Generate response using tools
    2. Critique: is the answer good enough?
    3. If good → END, else → regenerate with feedback
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tools: List[BaseTool],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        critique_prompt: str = DEFAULT_CRITIQUE_PROMPT,
        max_reflections: int = 3,
        max_tool_calls: int = 5,
        verbose: Literal["silent", "minimal", "normal", "verbose"] = "normal"
    ):
        self.config = config
        self.tools = tools
        self.system_prompt = system_prompt
        self.critique_prompt = critique_prompt
        self.max_reflections = max_reflections
        self.max_tool_calls = max_tool_calls
        self.verbose = VerboseLevel(verbose)

        self.llm = create_llm(config)
        self.llm_with_tools = self.llm.bind_tools(tools)

        self._last_run = {}
        self.graph = self._build_graph()

    def get_last_run(self) -> Dict[str, Any]:
        return self._last_run.copy()

    def _log(self, message: str, level: VerboseLevel = VerboseLevel.NORMAL):
        level_order = [VerboseLevel.SILENT, VerboseLevel.MINIMAL, VerboseLevel.NORMAL, VerboseLevel.VERBOSE]
        if level_order.index(self.verbose) >= level_order.index(level):
            print(message)

    def _build_graph(self) -> StateGraph:
        """Graph: generate ↔ tools → critique → END or back to generate"""
        workflow = StateGraph(ReflectionState)

        workflow.add_node("generate", self._generate_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("critique", self._critique_node)

        workflow.set_entry_point("generate")
        workflow.add_conditional_edges(
            "generate",
            self._after_generate,
            {"tools": "tools", "critique": "critique"}
        )
        workflow.add_edge("tools", "generate")
        workflow.add_conditional_edges(
            "critique",
            self._after_critique,
            {"generate": "generate", "end": END}
        )

        return workflow.compile()

    def _generate_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Generate response, using tools if needed"""
        messages = list(state["messages"])
        iteration = state.get("iteration", 0)
        critique = state.get("critique")

        # If coming back from critique (iteration > 0), keep tool data but fresh reasoning
        if iteration > 0 and critique:
            self._log(f"  [Gen {iteration+1}] Nouvelle tentative avec feedback...", VerboseLevel.NORMAL)

            # Keep only tool-related messages (tool calls + tool results)
            tool_messages = self._extract_tool_messages(messages)

            system_content = self.system_prompt + f"\n\nFeedback sur ta réponse précédente: {critique}\nAméliore ta réponse en tenant compte de ce feedback."
            messages = [
                {"role": "system", "content": system_content},
                HumanMessage(content=state["question"]),
                *tool_messages  # Include previous tool data
            ]
            response = self.llm_with_tools.invoke(messages)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                self._log(f"    Tools: {[tc['name'] for tc in response.tool_calls]}", VerboseLevel.NORMAL)
                return {"messages": [response]}

            return {"messages": [response], "current_response": response.content}

        # Count tool calls to prevent infinite loops
        tool_call_count = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls
        )

        # Force final answer if too many tool calls
        if tool_call_count >= self.max_tool_calls:
            response = self.llm.invoke([
                {"role": "system", "content": "Fournis ta meilleure réponse avec les informations collectées."},
                *messages
            ])
            self._log(f"  [Gen {iteration+1}] Réponse (max tools atteint)", VerboseLevel.NORMAL)
            return {"messages": [response], "current_response": response.content}

        # First generation: add system prompt
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [{"role": "system", "content": self.system_prompt}, messages[0]]

        response = self.llm_with_tools.invoke(messages)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            self._log(f"  [Gen {iteration+1}] Tools: {[tc['name'] for tc in response.tool_calls]}", VerboseLevel.NORMAL)
            return {"messages": [response]}

        self._log(f"  [Gen {iteration+1}] Réponse générée", VerboseLevel.NORMAL)
        return {"messages": [response], "current_response": response.content}

    def _after_generate(self, state: ReflectionState) -> str:
        """Route after generate: use tools or go to critique"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "critique"

    def _critique_node(self, state: ReflectionState) -> Dict[str, Any]:
        """Critique the response: is it good enough?"""
        iteration = state.get("iteration", 0)
        self._log(f"  [Critique {iteration+1}] Évaluation...", VerboseLevel.NORMAL)

        critique_prompt = self.critique_prompt.format(
            question=state.get("question", ""),
            response=state.get("current_response", "")
        )
        critique_response = self.llm.invoke(critique_prompt).content

        self._log(f"    {critique_response[:80]}...", VerboseLevel.VERBOSE)
        return {"critique": critique_response, "iteration": iteration + 1}

    def _after_critique(self, state: ReflectionState) -> str:
        """Decide: END if good, or regenerate with feedback"""
        iteration = state.get("iteration", 1)
        critique = state.get("critique", "")

        # Check if max reflections reached
        if iteration >= self.max_reflections:
            self._log(f"  → Max réflexions atteint, fin", VerboseLevel.NORMAL)
            return "end"

        # Parse critique to check if response is good
        is_good = self._parse_critique(critique)

        if is_good:
            self._log(f"  → Réponse validée!", VerboseLevel.NORMAL)
            return "end"
        else:
            self._log(f"  → Amélioration nécessaire, nouvelle génération...", VerboseLevel.NORMAL)
            return "generate"

    def _parse_critique(self, critique: str) -> bool:
        """Parse critique JSON to determine if response is good"""
        import json
        try:
            # Try to extract JSON from response
            critique_lower = critique.lower()
            if '"is_good": true' in critique_lower or '"is_good":true' in critique_lower:
                return True
            if '"is_good": false' in critique_lower or '"is_good":false' in critique_lower:
                return False
            # Fallback: try JSON parse
            data = json.loads(critique)
            return data.get("is_good", False)
        except:
            # If can't parse, assume needs improvement
            return False

    def _extract_tool_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Extract tool calls and tool results, discard AI text responses"""
        tool_messages = []
        for msg in messages:
            # Keep AI messages that have tool calls
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_messages.append(msg)
            # Keep tool results
            elif isinstance(msg, ToolMessage):
                tool_messages.append(msg)
            # Discard: HumanMessage (will re-add), AIMessage without tool calls (text responses)
        return tool_messages

    def invoke(self, question: str) -> str:
        self._log(f"\n{'='*50}", VerboseLevel.NORMAL)
        self._log(f"ReflectionAgent", VerboseLevel.NORMAL)
        self._log(f"{'='*50}", VerboseLevel.NORMAL)
        self._log(f"Question: {question[:80]}...", VerboseLevel.NORMAL)

        initial_state: ReflectionState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "current_response": None,
            "critique": None,
            "iteration": 0
        }

        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.max_reflections * self.max_tool_calls * 2 + 10}
        )

        self._last_run = {
            "question": question,
            "response": result.get("current_response"),
            "critique": result.get("critique"),
            "iterations": result.get("iteration", 0)
        }

        self._log(f"{'='*50}\n", VerboseLevel.NORMAL)
        return result.get("current_response") or "No response"

    def display_graph(self, return_image: bool = False):
        return display_graph(self.graph, return_image=return_image)

    def __repr__(self) -> str:
        return f"ReflectionAgent(tools={[t.name for t in self.tools]}, max_reflections={self.max_reflections})"
