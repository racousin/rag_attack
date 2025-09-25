"""Agent implementations for different RAG patterns"""

from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END


class TaskStatus(Enum):
    """Status of a task in the planner"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(TypedDict):
    """Task structure for planner agent"""
    id: str
    description: str
    status: TaskStatus
    dependencies: List[str]
    result: Optional[Any]


class PlannerState(TypedDict):
    """State for planner agent"""
    messages: List[BaseMessage]
    todo_list: List[Task]
    current_task: Optional[Task]
    completed_tasks: List[Task]
    context: Dict[str, Any]
    should_replan: bool


class RouterState(TypedDict):
    """State for router agent"""
    messages: List[BaseMessage]
    intent: str
    selected_tool: Optional[str]
    tool_result: Optional[Any]
    context: Dict[str, Any]


def create_planner_prompt(query: str, context: Optional[str] = None) -> str:
    """Create a prompt for task planning"""

    base_prompt = f"""You are a planning agent. Break down the following query into actionable tasks.

Query: {query}
"""

    if context:
        base_prompt += f"\nContext: {context}\n"

    base_prompt += """
Please provide a step-by-step plan with:
1. Clear, actionable tasks
2. Dependencies between tasks (if any)
3. Expected outcomes for each task

Format your response as a numbered list."""

    return base_prompt


def create_router_prompt(query: str, available_tools: List[str]) -> str:
    """Create a prompt for routing to appropriate tools"""

    tools_str = "\n".join([f"- {tool}" for tool in available_tools])

    prompt = f"""You are a routing agent. Analyze the user query and determine which tool to use.

Query: {query}

Available tools:
{tools_str}

Analyze the query and respond with:
1. The intent of the query
2. The most appropriate tool to use
3. Why you selected this tool

If no tool is appropriate, respond with "none" as the tool."""

    return prompt


class PlannerAgent:
    """Agent that creates and manages execution plans"""

    def __init__(self, llm):
        self.llm = llm
        self.task_counter = 0

    def generate_plan(self, state: PlannerState) -> PlannerState:
        """Generate a plan based on the query"""
        last_message = state["messages"][-1].content if state["messages"] else ""

        # Generate plan using LLM
        prompt = create_planner_prompt(last_message, state.get("context"))
        response = self.llm.invoke(prompt)

        # Parse response into tasks
        tasks = self._parse_plan(response.content)

        state["todo_list"] = tasks
        state["messages"].append(AIMessage(content=f"Generated plan with {len(tasks)} tasks"))

        return state

    def _parse_plan(self, plan_text: str) -> List[Task]:
        """Parse plan text into tasks"""
        tasks = []
        lines = plan_text.strip().split("\n")

        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:3]):
                self.task_counter += 1
                task = Task(
                    id=f"task_{self.task_counter}",
                    description=line.strip(),
                    status=TaskStatus.PENDING,
                    dependencies=[],
                    result=None
                )
                tasks.append(task)

        return tasks

    def execute_task(self, state: PlannerState) -> PlannerState:
        """Execute the current task"""
        if not state["current_task"]:
            # Select next pending task
            for task in state["todo_list"]:
                if task["status"] == TaskStatus.PENDING:
                    state["current_task"] = task
                    task["status"] = TaskStatus.IN_PROGRESS
                    break

        if state["current_task"]:
            # Simulate task execution (would call actual tools here)
            task = state["current_task"]

            # Mark as completed
            task["status"] = TaskStatus.COMPLETED
            task["result"] = f"Completed: {task['description']}"

            state["completed_tasks"].append(task)
            state["current_task"] = None

            state["messages"].append(
                AIMessage(content=f"Task completed: {task['description']}")
            )

        return state

    def should_replan(self, state: PlannerState) -> bool:
        """Determine if replanning is needed"""
        # Check for failed tasks or explicit replan flag
        failed_tasks = [t for t in state["todo_list"] if t["status"] == TaskStatus.FAILED]
        return len(failed_tasks) > 0 or state.get("should_replan", False)


class RouterAgent:
    """Agent that routes queries to appropriate tools"""

    def __init__(self, llm, tool_registry: Dict[str, Any]):
        self.llm = llm
        self.tool_registry = tool_registry

    def classify_intent(self, state: RouterState) -> RouterState:
        """Classify the intent of the query"""
        last_message = state["messages"][-1].content if state["messages"] else ""

        # Create routing prompt
        available_tools = list(self.tool_registry.keys())
        prompt = create_router_prompt(last_message, available_tools)

        # Get LLM response
        response = self.llm.invoke(prompt)

        # Parse intent and tool selection
        intent, tool = self._parse_routing_response(response.content)

        state["intent"] = intent
        state["selected_tool"] = tool
        state["messages"].append(
            AIMessage(content=f"Intent: {intent}, Selected tool: {tool}")
        )

        return state

    def _parse_routing_response(self, response: str) -> tuple[str, str]:
        """Parse the routing response to extract intent and tool"""
        # Simple parsing - can be enhanced
        lines = response.lower().split("\n")

        intent = "general"
        tool = "none"

        for line in lines:
            if "intent" in line:
                intent = line.split(":")[-1].strip()
            elif "tool" in line and "appropriate tool" in line:
                tool = line.split(":")[-1].strip()

        return intent, tool

    def execute_tool(self, state: RouterState) -> RouterState:
        """Execute the selected tool"""
        tool_name = state.get("selected_tool")

        if tool_name and tool_name != "none" and tool_name in self.tool_registry:
            tool = self.tool_registry[tool_name]

            # Execute tool with last message as input
            last_message = state["messages"][-1].content if state["messages"] else ""

            try:
                result = tool(last_message)
                state["tool_result"] = result
                state["messages"].append(
                    AIMessage(content=f"Tool execution result: {result}")
                )
            except Exception as e:
                state["tool_result"] = f"Error: {str(e)}"
                state["messages"].append(
                    AIMessage(content=f"Tool execution failed: {str(e)}")
                )
        else:
            state["messages"].append(
                AIMessage(content="No appropriate tool found for this query")
            )

        return state


def create_planner_graph(llm) -> StateGraph:
    """Create a planner agent graph"""
    planner = PlannerAgent(llm)

    graph = StateGraph(PlannerState)

    # Add nodes
    graph.add_node("plan", planner.generate_plan)
    graph.add_node("execute", planner.execute_task)

    # Set entry point
    graph.set_entry_point("plan")

    # Add edges
    def should_continue(state):
        # Check if all tasks are completed
        pending_tasks = [t for t in state["todo_list"] if t["status"] == TaskStatus.PENDING]

        if pending_tasks:
            return "execute"
        elif planner.should_replan(state):
            return "plan"
        else:
            return END

    graph.add_conditional_edges(
        "execute",
        should_continue,
        {
            "execute": "execute",
            "plan": "plan",
            END: END
        }
    )

    graph.add_edge("plan", "execute")

    return graph.compile()


def create_router_graph(llm, tool_registry: Dict[str, Any]) -> StateGraph:
    """Create a router agent graph"""
    router = RouterAgent(llm, tool_registry)

    graph = StateGraph(RouterState)

    # Add nodes
    graph.add_node("classify", router.classify_intent)
    graph.add_node("execute", router.execute_tool)

    # Set entry point
    graph.set_entry_point("classify")

    # Add edges
    graph.add_edge("classify", "execute")
    graph.add_edge("execute", END)

    return graph.compile()