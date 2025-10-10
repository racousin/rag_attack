"""SQL Database tool for querying Azure SQL"""
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import pyodbc


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


def sql_query_tool(config: Dict[str, Any], query: str, max_rows: int = 10) -> str:
    """
    Execute SQL queries against Azure SQL Database.

    Args:
        config: Azure configuration dictionary with SQL credentials
        query: The SQL query to execute
        max_rows: Maximum number of rows to return (default: 10)

    Returns:
        Query results as formatted string
    """
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


def get_database_schema(config: Dict[str, Any]) -> str:
    """
    Get the schema information for all tables in the database.

    Args:
        config: Azure configuration dictionary with SQL credentials

    Returns:
        Formatted schema information
    """
    try:
        conn = _get_sql_connection(config)

        try:
            # Query to get all tables and their columns
            schema_query = """
            SELECT
                t.TABLE_SCHEMA,
                t.TABLE_NAME,
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                c.COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.TABLES t
            JOIN INFORMATION_SCHEMA.COLUMNS c
                ON t.TABLE_SCHEMA = c.TABLE_SCHEMA
                AND t.TABLE_NAME = c.TABLE_NAME
            WHERE t.TABLE_TYPE = 'BASE TABLE'
            ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION
            """

            df = pd.read_sql(schema_query, conn)

            # Format schema information
            schema_info = []
            for table_name, group in df.groupby('TABLE_NAME'):
                schema_info.append(f"\nTable: {table_name}")
                schema_info.append("-" * 50)
                for _, row in group.iterrows():
                    nullable = "NULL" if row['IS_NULLABLE'] == 'YES' else "NOT NULL"
                    schema_info.append(
                        f"  {row['COLUMN_NAME']}: {row['DATA_TYPE']} {nullable}"
                    )

            return "\n".join(schema_info)

        finally:
            conn.close()

    except Exception as e:
        return f"Error getting database schema: {str(e)}"


def sql_table_info(config: Dict[str, Any], table_name: str) -> str:
    """
    Get detailed information about a specific table.

    Args:
        config: Azure configuration dictionary with SQL credentials
        table_name: Name of the table to inspect

    Returns:
        Table schema and sample data
    """
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
    from functools import partial

    return [
        partial(sql_query_tool, config),
        partial(get_database_schema, config),
        partial(sql_table_info, config)
    ]