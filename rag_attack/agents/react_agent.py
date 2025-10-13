"""ReAct (Reasoning and Acting) agent implementation"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import ReActSingleInputOutputParser
import operator
from ..agents.base_agent import create_llm
from ..utils.config import get_config


class ReActState(TypedDict):
    """State for ReAct agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    reasoning: List[str]
    observations: List[str]
    next_action: Optional[str]
    final_answer: Optional[str]


class ReActAgent:
    """
    ReAct agent that follows the Reasoning-Acting-Observation loop.

    This agent explicitly reasons about the task before taking actions,
    making its thought process transparent and debuggable.
    """

    def __init__(self, tools: List[BaseTool], max_iterations: int = 5, verbose: bool = True):
        """
        Initialize ReAct agent.

        Args:
            tools: List of tools available to the agent
            max_iterations: Maximum number of reasoning-acting cycles
            verbose: Whether to print progress information (default: True)
        """
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Get global config and create LLM
        config = get_config()
        self.llm = create_llm(config, temperature=0.0)

        # ReAct specific prompt
        self.react_prompt = """You are an assistant that uses the ReAct (Reasoning and Acting) framework to solve problems.

For each step:
1. Thought: Reason about what you need to do next
2. Action: Decide which tool to use and with what input
3. Observation: Observe the result of the action
4. Repeat until you have enough information to provide a final answer

Available tools:
{tool_descriptions}

Current task: {input}

Let's think step by step.

{agent_scratchpad}"""

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the ReAct workflow graph"""
        workflow = StateGraph(ReActState)

        # Add nodes
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("reason")

        # Add conditional edges
        workflow.add_conditional_edges(
            "reason",
            self._should_act_or_finalize,
            {
                "act": "act",
                "finalize": "finalize"
            }
        )

        workflow.add_edge("act", "observe")
        workflow.add_edge("observe", "reflect")

        workflow.add_conditional_edges(
            "reflect",
            self._should_continue_or_finalize,
            {
                "continue": "reason",
                "finalize": "finalize"
            }
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _reason_node(self, state: ReActState) -> Dict[str, Any]:
        """Reasoning step - think about what to do next"""
        messages = state.get("messages", [])
        reasoning = state.get("reasoning", [])
        observations = state.get("observations", [])

        if self.verbose:
            step_num = len(reasoning) + 1
            print(f"\n{'='*60}")
            print(f"🧠 STEP {step_num}: REASONING")
            print(f"{'='*60}")

        # Build context from previous reasoning and observations
        context = "\n".join([
            f"Thought {i+1}: {r}\nObservation {i+1}: {o}"
            for i, (r, o) in enumerate(zip(reasoning, observations))
        ])

        # Create reasoning prompt
        prompt = f"""Based on the current task and observations so far, what should I do next?

Task: {messages[0].content if messages else "No task specified"}

Previous reasoning and observations:
{context if context else "None yet"}

Current thought:"""

        if self.verbose:
            print("⏳ Thinking...")

        # Get reasoning from LLM
        response = self.llm.invoke(prompt)
        new_reasoning = response.content

        if self.verbose:
            print(f"\n💭 Thought: {new_reasoning}")

        # Determine next action
        if "final answer" in new_reasoning.lower() or len(reasoning) >= self.max_iterations:
            next_action = "finalize"
        else:
            next_action = "act"

        return {
            "reasoning": [new_reasoning],
            "next_action": next_action
        }

    def _act_node(self, state: ReActState) -> Dict[str, Any]:
        """Action step - execute a tool based on reasoning"""
        reasoning = state.get("reasoning", [])
        last_thought = reasoning[-1] if reasoning else ""

        if self.verbose:
            print(f"\n🔧 ACTION")
            print(f"{'-'*60}")

        # Parse the thought to determine which tool to use
        tool_prompt = f"""Based on this thought: "{last_thought}"

Which tool should I use and with what input? Available tools:
{self._get_tool_descriptions()}

Respond in this format:
Tool: [tool_name]
Input: [tool_input]"""

        if self.verbose:
            print("⏳ Deciding which tool to use...")

        response = self.llm.invoke(tool_prompt)

        # Parse tool and input
        lines = response.content.strip().split("\n")
        tool_name = ""
        tool_input = ""

        for line in lines:
            if line.startswith("Tool:"):
                tool_name = line.replace("Tool:", "").strip()
            elif line.startswith("Input:"):
                tool_input = line.replace("Input:", "").strip()

        if self.verbose:
            print(f"🔨 Using tool: {tool_name}")
            print(f"📝 Input: {tool_input[:100]}{'...' if len(tool_input) > 100 else ''}")
            print("⏳ Executing...")

        # Execute the tool
        tool_result = self._execute_tool(tool_name, tool_input)

        if self.verbose:
            result_preview = tool_result[:200] if len(tool_result) > 200 else tool_result
            print(f"\n✅ Result: {result_preview}{'...' if len(tool_result) > 200 else ''}")

        return {
            "observations": [f"Used {tool_name} with input '{tool_input}': {tool_result}"]
        }

    def _observe_node(self, state: ReActState) -> Dict[str, Any]:
        """Observation step - process tool output"""
        # Observations are already added in act_node
        # This node could be used for additional processing
        return {}

    def _reflect_node(self, state: ReActState) -> Dict[str, Any]:
        """Reflection step - evaluate progress and decide next steps"""
        reasoning = state.get("reasoning", [])
        observations = state.get("observations", [])
        messages = state.get("messages", [])

        if self.verbose:
            print(f"\n🤔 REFLECTION")
            print(f"{'-'*60}")
            print("⏳ Evaluating if we have enough information...")

        # Check if we have enough information
        reflection_prompt = f"""Based on the task and observations so far, do I have enough information to provide a final answer?

Task: {messages[0].content if messages else "No task"}

Reasoning and observations:
{self._format_history(reasoning, observations)}

Do I have enough information? (yes/no) and why:"""

        response = self.llm.invoke(reflection_prompt)

        # Decide whether to continue or finalize
        if "yes" in response.content.lower() or len(reasoning) >= self.max_iterations:
            if self.verbose:
                print("✅ Sufficient information gathered. Moving to finalize.")
            return {"next_action": "finalize"}
        else:
            if self.verbose:
                print("🔄 Need more information. Continuing reasoning cycle...")
            return {"next_action": "continue"}

    def _finalize_node(self, state: ReActState) -> Dict[str, Any]:
        """Finalize the answer based on all reasoning and observations"""
        reasoning = state.get("reasoning", [])
        observations = state.get("observations", [])
        messages = state.get("messages", [])

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 FINALIZING ANSWER")
            print(f"{'='*60}")
            print("⏳ Generating final answer based on all gathered information...")

        # Generate final answer
        final_prompt = f"""Based on all the reasoning and observations, provide a final answer to the task.

Task: {messages[0].content if messages else "No task"}

Reasoning and observations:
{self._format_history(reasoning, observations)}

Final answer:"""

        response = self.llm.invoke(final_prompt)

        if self.verbose:
            print(f"\n✨ FINAL ANSWER:")
            print(f"{'-'*60}")
            print(response.content)
            print(f"{'='*60}\n")

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)]
        }

    def _should_act_or_finalize(self, state: ReActState) -> str:
        """Determine whether to act or finalize"""
        next_action = state.get("next_action", "act")
        return next_action if next_action in ["act", "finalize"] else "act"

    def _should_continue_or_finalize(self, state: ReActState) -> str:
        """Determine whether to continue reasoning or finalize"""
        next_action = state.get("next_action", "continue")
        return next_action if next_action in ["continue", "finalize"] else "continue"

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by name with given input"""
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                try:
                    # Get the tool's input schema to determine the parameter name
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        # Get the first field name from the schema
                        schema_fields = tool.args_schema.model_fields
                        if schema_fields:
                            # Use the first parameter name
                            first_param = list(schema_fields.keys())[0]
                            return tool.invoke({first_param: tool_input})

                    # Fallback: try common parameter names
                    for param_name in ['query', 'input', 'text', 'question']:
                        try:
                            return tool.invoke({param_name: tool_input})
                        except:
                            continue

                    # Last resort: try with the input directly
                    return tool.invoke(tool_input)
                except Exception as e:
                    return f"Error executing tool: {str(e)}"

        return f"Tool '{tool_name}' not found"

    def _get_tool_descriptions(self) -> str:
        """Get descriptions of available tools"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

    def _format_history(self, reasoning: List[str], observations: List[str]) -> str:
        """Format reasoning and observations history"""
        history = []
        for i, (r, o) in enumerate(zip(reasoning, observations)):
            history.append(f"Step {i+1}:")
            history.append(f"  Thought: {r}")
            history.append(f"  Observation: {o}")
        return "\n".join(history)

    def invoke(self, question: str) -> str:
        """
        Invoke the ReAct agent with a question.

        Args:
            question: The question to answer

        Returns:
            The final answer
        """
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"🤖 ReAct Agent Starting")
            print(f"{'#'*60}")
            print(f"❓ Question: {question}")
            print(f"🔧 Available tools: {', '.join([tool.name for tool in self.tools])}")
            print(f"🔄 Max iterations: {self.max_iterations}")
            print(f"{'#'*60}")

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "reasoning": [],
            "observations": [],
            "next_action": None,
            "final_answer": None
        }

        result = self.graph.invoke(initial_state)
        return result.get("final_answer", "No answer generated")

    def get_reasoning_trace(self, question: str) -> Dict[str, Any]:
        """
        Get the full reasoning trace for a question.

        Args:
            question: The question to answer

        Returns:
            Dictionary containing the full reasoning trace
        """
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"🤖 ReAct Agent Starting (Trace Mode)")
            print(f"{'#'*60}")
            print(f"❓ Question: {question}")
            print(f"🔧 Available tools: {', '.join([tool.name for tool in self.tools])}")
            print(f"🔄 Max iterations: {self.max_iterations}")
            print(f"{'#'*60}")

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "reasoning": [],
            "observations": [],
            "next_action": None,
            "final_answer": None
        }

        # Collect all steps
        trace = []
        for step in self.graph.stream(initial_state):
            trace.append(step)

        # Get final state
        final_state = self.graph.invoke(initial_state)

        return {
            "question": question,
            "reasoning": final_state.get("reasoning", []),
            "observations": final_state.get("observations", []),
            "final_answer": final_state.get("final_answer", "No answer"),
            "trace": trace
        }