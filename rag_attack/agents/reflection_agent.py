"""Reflection Agent - Simple critique and improvement pattern"""
from typing import TypedDict, Annotated, Sequence, Dict, Any, Optional, Literal, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
import operator

from .simple_agent import create_llm, display_graph, VerboseLevel


class ReflectionState(TypedDict):
    """State for the reflection agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    initial_response: Optional[str]
    critique: Optional[str]
    final_response: Optional[str]


# Default prompts - students customize these
DEFAULT_PROMPTS = {
    "system": """Tu es un assistant intelligent qui utilise des outils pour répondre aux questions.
Utilise les outils disponibles pour obtenir des informations précises.
Réponds toujours dans la langue de la question.""",

    "critique": """Analyse cette réponse et identifie:
1. Les points forts
2. Les points à améliorer (clarté, complétude, précision)
3. Les informations manquantes

Question: {question}
Réponse: {response}

Critique:""",

    "improve": """Améliore cette réponse en tenant compte de la critique.

Question: {question}
Réponse initiale: {response}
Critique: {critique}

Réponse améliorée:"""
}


class ReflectionAgent:
    """
    Agent with reflection: generate → critique → improve → END

    Students can customize the 3 prompts to change behavior.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tools: List[BaseTool],
        prompts: Optional[Dict[str, str]] = None,
        max_iterations: int = 5,
        verbose: Literal["silent", "minimal", "normal", "verbose"] = "normal"
    ):
        self.config = config
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = VerboseLevel(verbose)

        self.llm = create_llm(config)
        self.llm_with_tools = self.llm.bind_tools(tools)

        # Prompts - merge defaults with user-provided
        self.prompts = {**DEFAULT_PROMPTS, **(prompts or {})}
        self._last_run = {}
        self.graph = self._build_graph()

    def get_prompts(self) -> Dict[str, str]:
        return self.prompts.copy()

    def set_prompts(self, prompts: Dict[str, str]):
        self.prompts.update(prompts)

    def get_last_run(self) -> Dict[str, Any]:
        return self._last_run.copy()

    def _log(self, message: str, level: VerboseLevel = VerboseLevel.NORMAL):
        level_order = [VerboseLevel.SILENT, VerboseLevel.MINIMAL, VerboseLevel.NORMAL, VerboseLevel.VERBOSE]
        if level_order.index(self.verbose) >= level_order.index(level):
            print(message)

    def _build_graph(self) -> StateGraph:
        """Fixed graph: generate → [tools] → critique → improve → END"""
        workflow = StateGraph(ReflectionState)

        workflow.add_node("generate", self._generate_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("improve", self._improve_node)

        workflow.set_entry_point("generate")
        workflow.add_conditional_edges(
            "generate",
            self._should_use_tools,
            {"tools": "tools", "critique": "critique"}
        )
        workflow.add_edge("tools", "generate")
        workflow.add_edge("critique", "improve")
        workflow.add_edge("improve", END)

        return workflow.compile()

    def _generate_node(self, state: ReflectionState) -> Dict[str, Any]:
        messages = state["messages"]
        tool_call_count = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls
        )

        if tool_call_count >= self.max_iterations:
            response = self.llm.invoke([
                {"role": "system", "content": "Fournis une réponse avec les informations collectées."},
                *messages
            ])
            return {"messages": [response], "initial_response": response.content}

        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [{"role": "system", "content": self.prompts["system"]}, messages[0]]

        response = self.llm_with_tools.invoke(messages)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            self._log(f"    Tools: {[tc['name'] for tc in response.tool_calls]}", VerboseLevel.NORMAL)
            return {"messages": [response]}
        return {"messages": [response], "initial_response": response.content}

    def _should_use_tools(self, state: ReflectionState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "critique"

    def _critique_node(self, state: ReflectionState) -> Dict[str, Any]:
        self._log(f"  [2/3] Critique...", VerboseLevel.NORMAL)
        critique_prompt = self.prompts["critique"].format(
            question=state.get("question", ""),
            response=state.get("initial_response", "")
        )
        critique = self.llm.invoke(critique_prompt).content
        self._log(f"    {critique[:100]}...", VerboseLevel.VERBOSE)
        return {"critique": critique}

    def _improve_node(self, state: ReflectionState) -> Dict[str, Any]:
        self._log(f"  [3/3] Amélioration...", VerboseLevel.NORMAL)
        improve_prompt = self.prompts["improve"].format(
            question=state.get("question", ""),
            response=state.get("initial_response", ""),
            critique=state.get("critique", "")
        )
        final = self.llm.invoke(improve_prompt).content
        return {"final_response": final}

    def invoke(self, question: str) -> str:
        self._log(f"\n{'='*50}", VerboseLevel.NORMAL)
        self._log(f"ReflectionAgent", VerboseLevel.NORMAL)
        self._log(f"{'='*50}", VerboseLevel.NORMAL)
        self._log(f"Question: {question[:80]}...", VerboseLevel.NORMAL)
        self._log(f"\n  [1/3] Génération initiale...", VerboseLevel.NORMAL)

        initial_state: ReflectionState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "initial_response": None,
            "critique": None,
            "final_response": None
        }

        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.max_iterations * 3 + 15}
        )

        self._last_run = {
            "question": question,
            "initial_response": result.get("initial_response"),
            "critique": result.get("critique"),
            "final_response": result.get("final_response")
        }

        self._log(f"{'='*50}\n", VerboseLevel.NORMAL)
        return result.get("final_response") or result.get("initial_response") or "No response"

    def display_graph(self, return_image: bool = False):
        return display_graph(self.graph, return_image=return_image)

    def __repr__(self) -> str:
        return f"ReflectionAgent(tools={[t.name for t in self.tools]})"
