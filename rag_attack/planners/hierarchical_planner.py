"""Hierarchical planner for complex multi-step tasks"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
import operator
import json
from ..agents.base_agent import create_llm
from ..utils.config import get_config


class PlanStep(TypedDict):
    """A single step in the plan"""
    step_id: int
    description: str
    dependencies: List[int]
    status: str  # "pending", "in_progress", "completed", "failed"
    result: Optional[str]
    tool: Optional[str]


class PlannerState(TypedDict):
    """State for the hierarchical planner"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    objective: str
    plan: List[PlanStep]
    current_step: Optional[int]
    execution_history: List[Dict[str, Any]]
    final_result: Optional[str]


class HierarchicalPlanner:
    """
    A hierarchical planner that breaks down complex tasks into subtasks,
    manages dependencies, and orchestrates execution.
    """

    def __init__(self, tools: List[BaseTool], max_steps: int = 10):
        """
        Initialize the hierarchical planner.

        Args:
            tools: List of available tools
            max_steps: Maximum number of steps in a plan
        """
        self.tools = tools
        self.max_steps = max_steps
        config = get_config()
        self.llm = create_llm(config, temperature=0.1)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the planner workflow graph"""
        workflow = StateGraph(PlannerState)

        # Add nodes
        workflow.add_node("analyze", self._analyze_objective)
        workflow.add_node("plan", self._create_plan)
        workflow.add_node("validate", self._validate_plan)
        workflow.add_node("execute_step", self._execute_step)
        workflow.add_node("evaluate", self._evaluate_progress)
        workflow.add_node("replan", self._replan)
        workflow.add_node("finalize", self._finalize)

        # Set entry point
        workflow.set_entry_point("analyze")

        # Add edges
        workflow.add_edge("analyze", "plan")
        workflow.add_edge("plan", "validate")

        workflow.add_conditional_edges(
            "validate",
            self._is_plan_valid,
            {
                "valid": "execute_step",
                "invalid": "plan"
            }
        )

        workflow.add_edge("execute_step", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "continue": "execute_step",
                "replan": "replan",
                "complete": "finalize"
            }
        )

        workflow.add_edge("replan", "validate")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _analyze_objective(self, state: PlannerState) -> Dict[str, Any]:
        """Analyze the objective to understand what needs to be done"""
        objective = state.get("objective", "")

        analysis_prompt = f"""Analyze this objective and identify the main components and requirements:

Objective: {objective}

Provide:
1. Main goal
2. Key requirements
3. Potential challenges
4. Success criteria"""

        response = self.llm.invoke(analysis_prompt)

        return {
            "messages": [AIMessage(content=f"Analysis: {response.content}")]
        }

    def _create_plan(self, state: PlannerState) -> Dict[str, Any]:
        """Create a detailed plan to achieve the objective"""
        objective = state.get("objective", "")
        messages = state.get("messages", [])

        # Get analysis if available
        analysis = ""
        for msg in messages:
            if isinstance(msg, AIMessage) and "Analysis:" in msg.content:
                analysis = msg.content
                break

        plan_prompt = f"""Create a detailed step-by-step plan to achieve this objective.

Objective: {objective}

{f"Analysis: {analysis}" if analysis else ""}

Available tools:
{self._get_tool_descriptions()}

Create a plan with the following format for each step:
{{
    "step_id": <number>,
    "description": "<what needs to be done>",
    "tool": "<tool_name or null>",
    "dependencies": [<list of step_ids this depends on>]
}}

Provide the plan as a JSON array of steps. Maximum {self.max_steps} steps."""

        response = self.llm.invoke(plan_prompt)

        # Parse the plan
        try:
            # Extract JSON from response
            content = response.content
            # Find JSON array in content
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                plan_json = json.loads(json_match.group())
            else:
                # Fallback: create a simple plan
                plan_json = [
                    {
                        "step_id": 1,
                        "description": "Execute the main task",
                        "tool": None,
                        "dependencies": []
                    }
                ]

            # Convert to PlanStep format
            plan = []
            for step in plan_json[:self.max_steps]:
                plan.append({
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "dependencies": step.get("dependencies", []),
                    "status": "pending",
                    "result": None,
                    "tool": step.get("tool")
                })

        except (json.JSONDecodeError, KeyError) as e:
            # Create a simple fallback plan
            plan = [{
                "step_id": 1,
                "description": objective,
                "dependencies": [],
                "status": "pending",
                "result": None,
                "tool": None
            }]

        return {"plan": plan}

    def _validate_plan(self, state: PlannerState) -> Dict[str, Any]:
        """Validate that the plan is executable"""
        plan = state.get("plan", [])

        # Check for circular dependencies
        for step in plan:
            if step["step_id"] in step["dependencies"]:
                return {"messages": [AIMessage(content="Invalid plan: circular dependency")]}

        # Check that dependencies exist
        step_ids = {step["step_id"] for step in plan}
        for step in plan:
            for dep in step["dependencies"]:
                if dep not in step_ids:
                    return {"messages": [AIMessage(content=f"Invalid plan: dependency {dep} not found")]}

        # Check tools exist
        tool_names = {tool.name for tool in self.tools}
        for step in plan:
            if step["tool"] and step["tool"] not in tool_names:
                return {"messages": [AIMessage(content=f"Invalid plan: tool {step['tool']} not found")]}

        return {"messages": [AIMessage(content="Plan validated successfully")]}

    def _execute_step(self, state: PlannerState) -> Dict[str, Any]:
        """Execute the next available step in the plan"""
        plan = state.get("plan", [])
        execution_history = state.get("execution_history", [])

        # Find next executable step
        next_step = None
        for step in plan:
            if step["status"] == "pending":
                # Check if dependencies are completed
                deps_completed = all(
                    any(s["step_id"] == dep and s["status"] == "completed" for s in plan)
                    for dep in step["dependencies"]
                )
                if deps_completed:
                    next_step = step
                    break

        if not next_step:
            return {"current_step": None}

        # Mark as in progress
        next_step["status"] = "in_progress"

        # Execute the step
        if next_step["tool"]:
            # Use specified tool
            tool_result = self._execute_tool_step(next_step, state)
            next_step["result"] = tool_result
        else:
            # Use LLM to complete the step
            llm_result = self._execute_llm_step(next_step, state)
            next_step["result"] = llm_result

        # Mark as completed
        next_step["status"] = "completed"

        # Add to execution history
        execution_history.append({
            "step_id": next_step["step_id"],
            "description": next_step["description"],
            "result": next_step["result"],
            "tool": next_step["tool"]
        })

        return {
            "plan": plan,
            "execution_history": execution_history,
            "current_step": next_step["step_id"]
        }

    def _execute_tool_step(self, step: PlanStep, state: PlannerState) -> str:
        """Execute a step using a specific tool"""
        tool_name = step["tool"]
        description = step["description"]

        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            return f"Tool {tool_name} not found"

        # Determine tool input from description and context
        input_prompt = f"""Based on this step description, what should be the input for the {tool_name} tool?

Step: {description}

Context from previous steps:
{self._format_execution_history(state.get("execution_history", []))}

Provide just the input value/query for the tool:"""

        response = self.llm.invoke(input_prompt)
        tool_input = response.content.strip()

        try:
            result = tool.invoke(tool_input)
            return f"Successfully executed {tool_name}: {result}"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _execute_llm_step(self, step: PlanStep, state: PlannerState) -> str:
        """Execute a step using the LLM"""
        description = step["description"]
        objective = state.get("objective", "")

        step_prompt = f"""Complete this step towards the objective:

Objective: {objective}
Current Step: {description}

Context from previous steps:
{self._format_execution_history(state.get("execution_history", []))}

Provide the result for this step:"""

        response = self.llm.invoke(step_prompt)
        return response.content

    def _evaluate_progress(self, state: PlannerState) -> Dict[str, Any]:
        """Evaluate progress and decide next action"""
        plan = state.get("plan", [])

        # Check if all steps are completed
        all_completed = all(step["status"] == "completed" for step in plan)
        if all_completed:
            return {"messages": [AIMessage(content="All steps completed")]}

        # Check for failed steps
        has_failed = any(step["status"] == "failed" for step in plan)
        if has_failed:
            return {"messages": [AIMessage(content="Some steps failed, replanning needed")]}

        # Continue execution
        pending_steps = sum(1 for step in plan if step["status"] == "pending")
        return {
            "messages": [AIMessage(content=f"{pending_steps} steps remaining")]
        }

    def _replan(self, state: PlannerState) -> Dict[str, Any]:
        """Replan when execution fails or gets stuck"""
        plan = state.get("plan", [])
        execution_history = state.get("execution_history", [])
        objective = state.get("objective", "")

        replan_prompt = f"""The current plan has issues. Create a revised plan.

Original objective: {objective}

Current plan status:
{self._format_plan_status(plan)}

Execution history:
{self._format_execution_history(execution_history)}

Create a new plan that addresses the issues and completes the remaining work."""

        # Use the create_plan logic
        state_copy = state.copy()
        state_copy["messages"].append(AIMessage(content=replan_prompt))
        return self._create_plan(state_copy)

    def _finalize(self, state: PlannerState) -> Dict[str, Any]:
        """Finalize the results and create summary"""
        objective = state.get("objective", "")
        plan = state.get("plan", [])
        execution_history = state.get("execution_history", [])

        summary_prompt = f"""Summarize the results of executing this plan:

Objective: {objective}

Execution results:
{self._format_execution_history(execution_history)}

Provide a concise summary of what was accomplished:"""

        response = self.llm.invoke(summary_prompt)

        return {
            "final_result": response.content,
            "messages": [AIMessage(content=response.content)]
        }

    def _is_plan_valid(self, state: PlannerState) -> str:
        """Check if the plan is valid"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if last_message and "Invalid plan" in last_message.content:
            return "invalid"
        return "valid"

    def _should_continue(self, state: PlannerState) -> str:
        """Determine next action based on evaluation"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if last_message:
            if "All steps completed" in last_message.content:
                return "complete"
            elif "failed" in last_message.content or "replanning" in last_message.content:
                return "replan"

        return "continue"

    def _get_tool_descriptions(self) -> str:
        """Get descriptions of available tools"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

    def _format_execution_history(self, history: List[Dict[str, Any]]) -> str:
        """Format execution history for context"""
        if not history:
            return "No previous steps executed"

        formatted = []
        for entry in history:
            formatted.append(
                f"Step {entry['step_id']}: {entry['description']}\n"
                f"Result: {entry.get('result', 'No result')[:200]}..."
            )
        return "\n\n".join(formatted)

    def _format_plan_status(self, plan: List[PlanStep]) -> str:
        """Format current plan status"""
        formatted = []
        for step in plan:
            status_emoji = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(step["status"], "❓")

            formatted.append(
                f"{status_emoji} Step {step['step_id']}: {step['description']} "
                f"(deps: {step['dependencies']})"
            )
        return "\n".join(formatted)

    def execute(self, objective: str) -> Dict[str, Any]:
        """
        Execute a complex objective using hierarchical planning.

        Args:
            objective: The high-level objective to achieve

        Returns:
            Dictionary containing the plan, execution history, and final result
        """
        initial_state = {
            "messages": [HumanMessage(content=objective)],
            "objective": objective,
            "plan": [],
            "current_step": None,
            "execution_history": [],
            "final_result": None
        }

        result = self.graph.invoke(initial_state)

        return {
            "objective": objective,
            "plan": result.get("plan", []),
            "execution_history": result.get("execution_history", []),
            "final_result": result.get("final_result", "No result generated")
        }