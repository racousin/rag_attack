"""ERP tool for querying VéloCorp SQL database"""
from typing import Optional
from langchain_core.tools import tool
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from ..utils.config import get_config


def _get_sql_engine():
    """Helper to get SQLAlchemy engine from config"""
    config = get_config()
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
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}")


@tool
def get_erp(query: str, max_rows: int = 10) -> str:
    """Execute SQL queries against VéloCorp ERP database (Azure SQL Server).

    IMPORTANT: Use T-SQL syntax (SQL Server):
    - Use TOP N instead of LIMIT N
    - Example: SELECT TOP 5 * FROM products ORDER BY price DESC

    DATABASE SCHEMA:

    Table: products
    - id, model, product_line, name, description, price, cost, stock_quantity
    - target_market ('B2C' or 'B2B'), color_options, size_options
    - weight_kg, material, warranty_months, launch_date, created_date

    Table: customers
    - id, customer_type ('B2C' or 'B2B'), segment, name, email, phone
    - address_street, address_city, address_postal_code, address_country
    - registration_date, loyalty_level, total_spent, order_count, last_order_date

    Table: orders
    - id, customer_id (FK), order_date, status ('en_attente', 'en_cours', 'expédié', 'livré', 'annulé')
    - channel, total_amount, discount_applied
    - shipping_street, shipping_city, shipping_postal_code, shipping_country
    - delivery_date, payment_method

    Table: order_items
    - id, order_id (FK), product_id (FK), quantity, unit_price, color, size

    Table: support_tickets
    - id, customer_id (FK), product_id (FK), category
    - priority ('basse', 'moyenne', 'haute', 'critique')
    - status ('ouvert', 'en_cours', 'résolu', 'fermé')
    - title, description, created_date, resolved_date, assigned_to, resolution_notes

    Table: invoices
    - id, order_id (FK), customer_id (FK), invoice_number, invoice_date, due_date
    - amount, tax_rate, tax_amount, total_amount
    - payment_status ('en_attente', 'payé', 'en_retard', 'annulé'), payment_date, payment_method

    Args:
        query: SQL SELECT query (T-SQL syntax)
        max_rows: Maximum rows to return (default: 10)

    Returns:
        Query results as formatted string

    Examples:
        get_erp("SELECT TOP 5 name, price FROM products ORDER BY price DESC")
        get_erp("SELECT name, total_spent FROM customers ORDER BY total_spent DESC", max_rows=10)
        get_erp("SELECT p.name, SUM(oi.quantity) as sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.name")
    """
    try:
        query_lower = query.lower().strip()

        # Must start with SELECT
        if not query_lower.startswith("select"):
            return "Error: Only SELECT queries are allowed for safety"

        # Block multiple statements (semicolon followed by another statement)
        if ';' in query_lower:
            # Check if there's anything meaningful after the semicolon
            parts = query_lower.split(';')
            for part in parts[1:]:
                if part.strip():  # Non-empty statement after semicolon
                    return "Error: Multiple SQL statements are not allowed"

        # Blacklist dangerous keywords
        dangerous_keywords = [
            'drop', 'delete', 'update', 'insert', 'alter', 'create',
            'truncate', 'exec', 'execute', 'grant', 'revoke',
            'merge', 'replace', 'rename', 'xp_', 'sp_'
        ]

        for keyword in dangerous_keywords:
            # Check for keyword as whole word (with word boundaries)
            # This prevents false positives like "updated_at" column
            import re
            if re.search(rf'\b{keyword}\b', query_lower):
                return f"Error: '{keyword.upper()}' is not allowed in queries"

        engine = _get_sql_engine()
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

            if len(df) > max_rows:
                df = df.head(max_rows)
                result_str = f"Showing first {max_rows} rows:\n\n"
            else:
                result_str = f"Query returned {len(df)} rows:\n\n"

            result_str += df.to_string(index=False)
            return result_str

    except Exception as e:
        return f"Error executing SQL query: {str(e)}"
