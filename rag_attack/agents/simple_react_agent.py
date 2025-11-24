"""Simplified ReAct agent that actually works"""
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.base_agent import create_llm
from ..utils.config import get_config


class SimpleReActAgent:
    """
    Simplified ReAct (Reasoning and Acting) agent.

    Demonstrates the ReAct pattern:
    - Thought: Explicit reasoning about what to do
    - Action: Execute a tool
    - Observation: See the result
    - Repeat until enough information gathered

    Uses simple heuristics for stopping instead of complex LLM-based reflection.
    """

    def __init__(self, tools: List[BaseTool], max_iterations: int = 5, verbose: bool = True):
        """
        Initialize SimpleReAct agent.

        Args:
            tools: List of tools available to the agent
            max_iterations: Maximum number of reasoning-acting cycles
            verbose: Whether to print progress information
        """
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose

        config = get_config()
        self.llm = create_llm(config, temperature=0.0)

        self.tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in tools
        ])

    def invoke(self, question: str) -> str:
        """
        Invoke the ReAct agent with a question.

        Args:
            question: The question to answer

        Returns:
            The final answer
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🤖 SimpleReAct Agent")
            print(f"{'='*70}")
            print(f"❓ Question: {question}")
            print(f"🔧 Tools: {', '.join([t.name for t in self.tools])}")
            print(f"🔄 Max iterations: {self.max_iterations}")
            print(f"{'='*70}\n")

        # Track history
        thoughts = []
        observations = []

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n{'─'*70}")
                print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
                print(f"{'─'*70}")

            # STEP 1: THOUGHT - Reason about what to do next
            thought = self._generate_thought(question, thoughts, observations, iteration)
            thoughts.append(thought)

            if self.verbose:
                print(f"\n💭 THOUGHT:")
                print(f"   {thought}")

            # Check if agent thinks it's done
            if self._should_finalize(thought, observations, iteration):
                if self.verbose:
                    print(f"\n✅ Agent has enough information. Generating final answer...")
                break

            # STEP 2: ACTION - Decide which tool to use and execute it
            action_result = self._execute_action(thought)

            if action_result["error"]:
                observations.append(f"Error: {action_result['error']}")
                if self.verbose:
                    print(f"\n❌ ACTION ERROR:")
                    print(f"   {action_result['error']}")
                continue

            # STEP 3: OBSERVATION - Record what we learned
            # Increase limit to capture more content (technical specs are often beyond 200 chars)
            observation = f"Used {action_result['tool']} and found: {action_result['result'][:1000]}..."
            observations.append(observation)

            if self.verbose:
                print(f"\n🔧 ACTION:")
                print(f"   Tool: {action_result['tool']}")
                print(f"   Input: {action_result['input']}")
                print(f"\n👁️  OBSERVATION:")
                print(f"   {observation}")

        # FINAL STEP: Generate answer based on all observations
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📝 GENERATING FINAL ANSWER")
            print(f"{'='*70}\n")

        final_answer = self._generate_final_answer(question, thoughts, observations)

        if self.verbose:
            print(f"✨ ANSWER:")
            print(f"{final_answer}")
            print(f"\n{'='*70}\n")

        return final_answer

    def _generate_thought(self, question: str, thoughts: List[str], observations: List[str], iteration: int) -> str:
        """Generate reasoning about what to do next"""

        history = ""
        if thoughts and observations:
            history = "\n\nPrevious iterations:"
            for i, (t, o) in enumerate(zip(thoughts, observations), 1):
                history += f"\n  Iteration {i}:"
                history += f"\n    Thought: {t[:100]}..."
                history += f"\n    Observation: {o[:100]}..."

        prompt = f"""You are solving this task: {question}

Available tools:
{self.tool_descriptions}
{history}

Current iteration: {iteration}/{self.max_iterations}

IMPORTANT:
- Use the SAME LANGUAGE as the question for your search queries
- If the question is in French, use French keywords. If in English, use English keywords
- Use SIMPLE, FOCUSED keywords (1-3 words) rather than complex phrases
- Example: Instead of "problèmes courants signalés par les clients concernant les freins", use "freins" or "problème frein"

Think step-by-step:
1. What information do I still need to answer the question?
2. Which tool should I use to get that information?
3. What SIMPLE, FOCUSED query should I provide? (Use the same language as the question!)

If you have gathered enough information from previous observations, state: "I have enough information to answer."

Your reasoning:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _should_finalize(self, thought: str, observations: List[str], iteration: int) -> bool:
        """Simple heuristics to decide if we should stop"""

        # Check if agent explicitly says it has enough
        stop_phrases = [
            "enough information",
            "i can now answer",
            "ready to answer",
            "sufficient information",
            "have all the information"
        ]
        if any(phrase in thought.lower() for phrase in stop_phrases):
            return True

        # Stop if we have 3+ successful observations
        if len([o for o in observations if not o.startswith("Error")]) >= 3:
            return True

        # Stop on last iteration
        if iteration >= self.max_iterations:
            return True

        return False

    def _execute_action(self, thought: str) -> Dict[str, Any]:
        """Parse thought and execute the appropriate tool"""

        # Ask LLM to extract tool and input from the thought
        prompt = f"""Based on this reasoning: "{thought}"

Available tools:
{self.tool_descriptions}

Extract the tool to use and the specific input for it.
CRITICAL: Keep the Input in the SAME LANGUAGE as the reasoning. Do NOT translate!

Respond in EXACTLY this format (no other text):
Tool: [tool_name]
Input: [specific query in the original language]

Example (French):
Tool: azure_search_tool
Input: vélos électriques E-City

Example (English):
Tool: azure_search_tool
Input: electric bikes E-City"""

        response = self.llm.invoke(prompt)
        content = response.content.strip()

        # Parse response
        tool_name = ""
        tool_input = ""

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Tool:"):
                tool_name = line.replace("Tool:", "").strip()
            elif line.startswith("Input:"):
                tool_input = line.replace("Input:", "").strip()

        if not tool_name or not tool_input:
            return {
                "error": f"Could not parse action. Tool: '{tool_name}', Input: '{tool_input}'",
                "tool": None,
                "input": None,
                "result": None
            }

        # Execute the tool
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                try:
                    # Try to invoke with the first parameter
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        schema_fields = tool.args_schema.model_fields
                        if schema_fields:
                            first_param = list(schema_fields.keys())[0]
                            result = tool.invoke({first_param: tool_input})
                            return {
                                "error": None,
                                "tool": tool_name,
                                "input": tool_input,
                                "result": result
                            }
                except Exception as e:
                    return {
                        "error": f"Tool execution failed: {str(e)}",
                        "tool": tool_name,
                        "input": tool_input,
                        "result": None
                    }

        return {
            "error": f"Tool '{tool_name}' not found",
            "tool": tool_name,
            "input": tool_input,
            "result": None
        }

    def _generate_final_answer(self, question: str, thoughts: List[str], observations: List[str]) -> str:
        """Generate final answer based on all reasoning and observations"""

        history = "\n\nGathered information:"
        for i, (t, o) in enumerate(zip(thoughts, observations), 1):
            history += f"\n\nIteration {i}:"
            history += f"\n  Reasoning: {t}"
            history += f"\n  Observation: {o}"

        prompt = f"""Question: {question}
{history}

CRITICAL INSTRUCTIONS:
1. Base your answer ONLY on the observations gathered above
2. If the observations contain relevant information, use it to answer the question
3. If the observations are empty or contain errors, clearly state that you couldn't find the information
4. DO NOT make up information that is not in the observations
5. Answer in the SAME LANGUAGE as the question

Your answer:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def get_reasoning_trace(self, question: str) -> Dict[str, Any]:
        """
        Get the full reasoning trace for a question.

        Returns:
            Dictionary with reasoning steps, observations, and final answer
        """
        # Temporarily disable verbose for cleaner trace
        original_verbose = self.verbose
        self.verbose = False

        thoughts = []
        observations = []

        for iteration in range(1, self.max_iterations + 1):
            thought = self._generate_thought(question, thoughts, observations, iteration)
            thoughts.append(thought)

            if self._should_finalize(thought, observations, iteration):
                break

            action_result = self._execute_action(thought)

            if action_result["error"]:
                observations.append(f"Error: {action_result['error']}")
            else:
                observation = f"Used {action_result['tool']}: {action_result['result'][:1000]}"
                observations.append(observation)

        final_answer = self._generate_final_answer(question, thoughts, observations)

        self.verbose = original_verbose

        return {
            "question": question,
            "reasoning": thoughts,
            "observations": observations,
            "final_answer": final_answer
        }
