"""SQL Database tool for querying Azure SQL"""
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import pyodbc

# Import config management from centralized location
from ..utils.config import get_config


class SQLQuery(BaseModel):
    """Input schema for SQL query tool"""
    query: str = Field(description="The SQL query to execute")
    max_rows: int = Field(default=10, description="Maximum number of rows to return")


def _get_sql_connection(config: Dict[str, Any]):
    """Helper to get SQL connection from config"""
    connection_string = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={config['sql_server']};"
        f"Database={config['sql_database']};"
        f"Uid={config['sql_username']};"
        f"Pwd={config['sql_password']};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )
    return pyodbc.connect(connection_string)


@tool
def sql_query_tool(query: str, max_rows: int = 10) -> str:
    """Execute SQL queries against Azure SQL Server Database (T-SQL syntax).

    IMPORTANT: This is SQL Server, use T-SQL syntax:
    - Use TOP N instead of LIMIT N
    - Example: SELECT TOP 5 * FROM table ORDER BY column DESC

    DATABASE SCHEMA:

    Table: products
    - id: INT PRIMARY KEY (auto-increment)
    - model: NVARCHAR(100) NOT NULL
    - product_line: NVARCHAR(50) NOT NULL
    - name: NVARCHAR(200) NOT NULL
    - description: NVARCHAR(MAX)
    - price: DECIMAL(10,2) NOT NULL
    - cost: DECIMAL(10,2) NOT NULL
    - stock_quantity: INT DEFAULT 0
    - target_market: NVARCHAR(10) ('B2C' or 'B2B')
    - color_options: NVARCHAR(200)
    - size_options: NVARCHAR(100)
    - weight_kg: DECIMAL(5,2)
    - material: NVARCHAR(50)
    - warranty_months: INT DEFAULT 12
    - launch_date: DATE
    - created_date: DATETIME2

    Table: customers
    - id: INT PRIMARY KEY (auto-increment)
    - customer_type: NVARCHAR(10) ('B2C' or 'B2B')
    - segment: NVARCHAR(50)
    - name: NVARCHAR(200) NOT NULL
    - email: NVARCHAR(200) NOT NULL
    - phone: NVARCHAR(50)
    - address_street: NVARCHAR(200)
    - address_city: NVARCHAR(100)
    - address_postal_code: NVARCHAR(20)
    - address_country: NVARCHAR(100)
    - registration_date: DATE
    - loyalty_level: NVARCHAR(20)
    - total_spent: DECIMAL(12,2) DEFAULT 0
    - order_count: INT DEFAULT 0
    - last_order_date: DATE
    - notes: NVARCHAR(MAX)
    - created_date: DATETIME2

    Table: orders
    - id: INT PRIMARY KEY (auto-increment)
    - customer_id: INT NOT NULL (FK to customers.id)
    - order_date: DATE NOT NULL
    - status: NVARCHAR(20) ('en_attente', 'en_cours', 'expédié', 'livré', 'annulé')
    - channel: NVARCHAR(20)
    - total_amount: DECIMAL(10,2) NOT NULL
    - discount_applied: DECIMAL(10,2) DEFAULT 0
    - shipping_street: NVARCHAR(200)
    - shipping_city: NVARCHAR(100)
    - shipping_postal_code: NVARCHAR(20)
    - shipping_country: NVARCHAR(100)
    - delivery_date: DATE
    - payment_method: NVARCHAR(30)
    - notes: NVARCHAR(MAX)
    - created_date: DATETIME2

    Table: order_items
    - id: INT PRIMARY KEY (auto-increment)
    - order_id: INT NOT NULL (FK to orders.id)
    - product_id: INT NOT NULL (FK to products.id)
    - quantity: INT NOT NULL DEFAULT 1
    - unit_price: DECIMAL(10,2) NOT NULL
    - color: NVARCHAR(50)
    - size: NVARCHAR(10)

    Table: support_tickets
    - id: INT PRIMARY KEY (auto-increment)
    - customer_id: INT NOT NULL (FK to customers.id)
    - product_id: INT (FK to products.id)
    - category: NVARCHAR(50)
    - priority: NVARCHAR(20) ('basse', 'moyenne', 'haute', 'critique')
    - status: NVARCHAR(20) ('ouvert', 'en_cours', 'résolu', 'fermé')
    - title: NVARCHAR(200) NOT NULL
    - description: NVARCHAR(MAX)
    - created_date: DATE NOT NULL
    - resolved_date: DATE
    - assigned_to: NVARCHAR(100)
    - resolution_notes: NVARCHAR(MAX)

    Table: invoices
    - id: INT PRIMARY KEY (auto-increment)
    - order_id: INT NOT NULL (FK to orders.id)
    - customer_id: INT NOT NULL (FK to customers.id)
    - invoice_number: NVARCHAR(50) UNIQUE NOT NULL
    - invoice_date: DATE NOT NULL
    - due_date: DATE NOT NULL
    - amount: DECIMAL(10,2) NOT NULL
    - tax_rate: DECIMAL(5,4) DEFAULT 0.20
    - tax_amount: DECIMAL(10,2)
    - total_amount: DECIMAL(10,2)
    - payment_status: NVARCHAR(20) ('en_attente', 'payé', 'en_retard', 'annulé')
    - payment_date: DATE
    - payment_method: NVARCHAR(30)
    - notes: NVARCHAR(MAX)

    Args:
        query: The SQL query to execute (SELECT only, T-SQL syntax)
        max_rows: Maximum number of rows to return (default: 10)

    Returns:
        Query results as formatted string
    """
    config = get_config()
    try:
        # Security check - only allow SELECT queries
        query_lower = query.lower().strip()
        if not query_lower.startswith("select"):
            return "Error: Only SELECT queries are allowed for safety"

        # Get connection
        conn = _get_sql_connection(config)

        try:
            # Execute query
            df = pd.read_sql(query, conn)

            # Limit rows
            if len(df) > max_rows:
                df = df.head(max_rows)
                result_str = f"Showing first {max_rows} rows of {len(df)} total:\n\n"
            else:
                result_str = f"Query returned {len(df)} rows:\n\n"

            # Format results
            result_str += df.to_string(index=False)

            return result_str

        finally:
            conn.close()

    except Exception as e:
        return f"Error executing SQL query: {str(e)}"


@tool
def sql_table_info(table_name: str) -> str:
    """Get detailed information about a specific table including sample data.

    Available tables: products, customers, orders, order_items, support_tickets, invoices

    Note: For full schema of all tables, check the sql_query_tool description which contains
    the complete database schema.

    Args:
        table_name: Name of the table to inspect (e.g., 'products', 'customers', 'orders')

    Returns:
        Table schema and sample data
    """
    config = get_config()
    try:
        conn = _get_sql_connection(config)

        try:
            # Get column information
            column_query = f"""
            SELECT
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table_name}'
            ORDER BY ORDINAL_POSITION
            """

            columns_df = pd.read_sql(column_query, conn)

            if columns_df.empty:
                return f"Table '{table_name}' not found"

            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_df = pd.read_sql(count_query, conn)
            row_count = count_df['count'].iloc[0]

            # Get sample data
            sample_query = f"SELECT TOP 5 * FROM {table_name}"
            sample_df = pd.read_sql(sample_query, conn)

            # Format output
            output = [f"Table: {table_name}"]
            output.append(f"Total rows: {row_count}")
            output.append("\nColumns:")
            output.append("-" * 50)

            for _, row in columns_df.iterrows():
                nullable = "NULL" if row['IS_NULLABLE'] == 'YES' else "NOT NULL"
                output.append(f"  {row['COLUMN_NAME']}: {row['DATA_TYPE']} {nullable}")

            output.append("\nSample data (first 5 rows):")
            output.append("-" * 50)
            output.append(sample_df.to_string(index=False))

            return "\n".join(output)

        finally:
            conn.close()

    except Exception as e:
        return f"Error getting table info: {str(e)}"


def create_sql_agent_tools(config: Dict[str, Any]):
    """
    Create a collection of SQL tools for the agent with config bound.

    Args:
        config: Azure configuration dictionary

    Returns:
        List of SQL-related tools with config bound
    """
    from functools import partial, update_wrapper

    # Create partial functions and set their __name__ attribute for LangChain compatibility
    sql_query_partial = partial(sql_query_tool, config)
    sql_query_partial.__name__ = "sql_query_tool"

    table_info_partial = partial(sql_table_info, config)
    table_info_partial.__name__ = "sql_table_info"

    return [
        sql_query_partial,
        table_info_partial
    ]


# ============================================================================
# CLEAN WRAPPER FUNCTIONS (No config parameter needed!)
# ============================================================================

def execute_sql_query(query: str, max_rows: int = 10) -> str:
    """
    Execute SQL queries against Azure SQL Database.
    Uses the global configuration set via set_config().

    Args:
        query: The SQL query to execute (SELECT only)
        max_rows: Maximum number of rows to return (default: 10)

    Returns:
        Query results as formatted string
    """
    return sql_query_tool.invoke({"query": query, "max_rows": max_rows})




def get_table_info(table_name: str) -> str:
    """
    Get detailed information about a specific table.
    Uses the global configuration set via set_config().

    Args:
        table_name: Name of the table to inspect

    Returns:
        Table schema and sample data
    """
    return sql_table_info.invoke({"table_name": table_name})