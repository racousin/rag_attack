"""Tools for RAG implementations including Traditional and MCP tools"""

from typing import Dict, List, Any, Optional, Callable, Protocol
from dataclasses import dataclass
import json
import asyncio
from enum import Enum


class ToolType(Enum):
    """Type of tool implementation"""
    TRADITIONAL = "traditional"
    MCP = "mcp"


class ToolCapability(Enum):
    """Capabilities that tools can provide"""
    SEARCH = "search"
    COMPUTE = "compute"
    DATABASE = "database"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"


@dataclass
class ToolSpec:
    """Specification for a tool"""
    name: str
    description: str
    type: ToolType
    capabilities: List[ToolCapability]
    parameters: Dict[str, Any]
    function: Optional[Callable] = None
    endpoint: Optional[str] = None


class Tool(Protocol):
    """Protocol for tool implementations"""

    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool"""
        ...

    def get_spec(self) -> ToolSpec:
        """Get tool specification"""
        ...


class TraditionalTool:
    """Traditional function-based tool"""

    def __init__(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters
        self.type = ToolType.TRADITIONAL

    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool function"""
        return self.function(*args, **kwargs)

    def get_spec(self) -> ToolSpec:
        """Get tool specification"""
        return ToolSpec(
            name=self.name,
            description=self.description,
            type=self.type,
            capabilities=self._infer_capabilities(),
            parameters=self.parameters,
            function=self.function
        )

    def _infer_capabilities(self) -> List[ToolCapability]:
        """Infer capabilities from tool name and description"""
        capabilities = []

        lower_desc = (self.name + " " + self.description).lower()

        if "search" in lower_desc or "query" in lower_desc:
            capabilities.append(ToolCapability.SEARCH)
        if "calculate" in lower_desc or "compute" in lower_desc:
            capabilities.append(ToolCapability.COMPUTE)
        if "database" in lower_desc or "sql" in lower_desc:
            capabilities.append(ToolCapability.DATABASE)
        if "retrieve" in lower_desc or "fetch" in lower_desc:
            capabilities.append(ToolCapability.RETRIEVAL)
        if "generate" in lower_desc or "create" in lower_desc:
            capabilities.append(ToolCapability.GENERATION)

        return capabilities if capabilities else [ToolCapability.COMPUTE]


class MCPTool:
    """Model Context Protocol tool"""

    def __init__(self, name: str, endpoint: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.endpoint = endpoint
        self.description = description
        self.parameters = parameters
        self.type = ToolType.MCP
        self._client = None

    async def execute_async(self, *args, **kwargs) -> Any:
        """Execute MCP tool asynchronously"""
        # Simulate MCP protocol communication
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": self.name,
                "arguments": kwargs
            },
            "id": 1
        }

        # In real implementation, would send to MCP server
        # For now, return simulated response
        return {
            "jsonrpc": "2.0",
            "result": {
                "output": f"MCP tool {self.name} executed with params: {kwargs}"
            },
            "id": 1
        }

    def execute(self, *args, **kwargs) -> Any:
        """Execute MCP tool synchronously"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_async(*args, **kwargs))

    def get_spec(self) -> ToolSpec:
        """Get tool specification"""
        return ToolSpec(
            name=self.name,
            description=self.description,
            type=self.type,
            capabilities=self._get_capabilities(),
            parameters=self.parameters,
            endpoint=self.endpoint
        )

    def _get_capabilities(self) -> List[ToolCapability]:
        """Get capabilities from MCP server"""
        # In real implementation, would query MCP server
        # For now, return inferred capabilities
        capabilities = []

        lower_desc = (self.name + " " + self.description).lower()

        if "search" in lower_desc:
            capabilities.append(ToolCapability.SEARCH)
        if "compute" in lower_desc:
            capabilities.append(ToolCapability.COMPUTE)
        if "database" in lower_desc:
            capabilities.append(ToolCapability.DATABASE)

        return capabilities if capabilities else [ToolCapability.COMPUTE]


class ToolRegistry:
    """Registry for managing tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.capabilities_map: Dict[ToolCapability, List[str]] = {}

    def register(self, tool: Tool):
        """Register a tool"""
        spec = tool.get_spec()
        self.tools[spec.name] = tool

        # Update capabilities map
        for capability in spec.capabilities:
            if capability not in self.capabilities_map:
                self.capabilities_map[capability] = []
            self.capabilities_map[capability].append(spec.name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def get_tools_by_capability(self, capability: ToolCapability) -> List[Tool]:
        """Get tools that have a specific capability"""
        tool_names = self.capabilities_map.get(capability, [])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())

    def get_tool_specs(self) -> List[ToolSpec]:
        """Get specifications for all tools"""
        return [tool.get_spec() for tool in self.tools.values()]


# Example tool implementations

def create_search_tool(search_endpoint: str, search_key: str) -> TraditionalTool:
    """Create a search tool"""

    def search_function(query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Search for documents"""
        # Simulated search - in real implementation would call Azure Search
        return [
            {"title": f"Document {i}", "content": f"Content about {query}"}
            for i in range(top_k)
        ]

    return TraditionalTool(
        name="search",
        function=search_function,
        description="Search for relevant documents",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results", "default": 5}
        }
    )


def create_calculator_tool() -> TraditionalTool:
    """Create a calculator tool"""

    def calculate(expression: str) -> float:
        """Evaluate a mathematical expression"""
        try:
            # Safely evaluate mathematical expression
            import ast
            import operator

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow
            }

            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                else:
                    raise TypeError(node)

            return eval_expr(ast.parse(expression, mode='eval').body)
        except:
            return 0.0

    return TraditionalTool(
        name="calculator",
        function=calculate,
        description="Calculate mathematical expressions",
        parameters={
            "expression": {"type": "string", "description": "Mathematical expression"}
        }
    )


def create_database_tool(connection_string: str) -> TraditionalTool:
    """Create a database query tool"""

    def query_database(query: str) -> List[Dict[str, Any]]:
        """Query the database"""
        # Simulated database query - in real implementation would use actual DB
        return [
            {"id": 1, "result": f"Result for query: {query}"}
        ]

    return TraditionalTool(
        name="database",
        function=query_database,
        description="Query SQL database",
        parameters={
            "query": {"type": "string", "description": "SQL query"}
        }
    )


def create_mcp_search_tool(endpoint: str) -> MCPTool:
    """Create an MCP-based search tool"""
    return MCPTool(
        name="mcp_search",
        endpoint=endpoint,
        description="Search using Model Context Protocol",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "filters": {"type": "object", "description": "Search filters"}
        }
    )


def compare_tool_types() -> Dict[str, Any]:
    """Compare traditional and MCP tools"""
    return {
        "Traditional Tools": {
            "advantages": [
                "Direct function execution",
                "No network overhead",
                "Simple integration",
                "Full control over implementation",
                "Synchronous by default"
            ],
            "disadvantages": [
                "Tightly coupled to codebase",
                "Harder to scale",
                "Language-specific",
                "Manual versioning"
            ],
            "use_cases": [
                "Local computations",
                "Simple transformations",
                "Performance-critical operations",
                "Prototype development"
            ]
        },
        "MCP Tools": {
            "advantages": [
                "Language agnostic",
                "Standardized protocol",
                "Tool discovery",
                "Version management",
                "Distributed architecture",
                "Dynamic capabilities"
            ],
            "disadvantages": [
                "Network overhead",
                "More complex setup",
                "Requires MCP server",
                "Asynchronous handling"
            ],
            "use_cases": [
                "Distributed systems",
                "Multi-language environments",
                "Dynamic tool loading",
                "Production deployments",
                "Microservices architecture"
            ]
        },
        "Decision Matrix": {
            "Use Traditional When": [
                "Building prototypes",
                "Need low latency",
                "Simple tool requirements",
                "Single language/framework"
            ],
            "Use MCP When": [
                "Building production systems",
                "Need tool discovery",
                "Multiple language support",
                "Distributed architecture",
                "Dynamic tool management"
            ]
        }
    }