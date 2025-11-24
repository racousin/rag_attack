"""Report writing tool for generating structured reports"""
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime


@tool
def report_writer_tool(
    title: str,
    content: str,
    report_type: str = "general",
    format: str = "markdown"
) -> str:
    """Generate a professional report with proper formatting.

    Args:
        title: Title of the report
        content: Main content/analysis for the report
        report_type: Type of report (general, technical, business, customer_analysis)
        format: Output format (markdown, text, html)

    Returns:
        Formatted report as a string
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if format == "markdown":
            report = _generate_markdown_report(title, content, report_type, timestamp)
        elif format == "html":
            report = _generate_html_report(title, content, report_type, timestamp)
        else:
            report = _generate_text_report(title, content, report_type, timestamp)

        return report

    except Exception as e:
        return f"Error generating report: {str(e)}"


def _generate_markdown_report(title: str, content: str, report_type: str, timestamp: str) -> str:
    """Generate a markdown formatted report"""

    template_sections = {
        "technical": ["## Résumé Exécutif", "## Analyse Technique", "## Problèmes Identifiés", "## Solutions Recommandées", "## Conclusion"],
        "business": ["## Résumé Exécutif", "## Contexte", "## Analyse", "## Opportunités", "## Recommandations", "## Prochaines Étapes"],
        "customer_analysis": ["## Résumé", "## Analyse des Retours Clients", "## Points Positifs", "## Points d'Amélioration", "## Plan d'Action"],
        "general": ["## Introduction", "## Contenu Principal", "## Conclusion"]
    }

    sections = template_sections.get(report_type, template_sections["general"])

    report = f"""# {title}

---
**Type de Rapport:** {report_type.replace('_', ' ').title()}
**Date de Génération:** {timestamp}
**Généré par:** RAG Agent System

---

{content}

---

## Métadonnées
- Type: {report_type}
- Format: Markdown
- Généré automatiquement par le système RAG agentique
"""

    return report


def _generate_html_report(title: str, content: str, report_type: str, timestamp: str) -> str:
    """Generate an HTML formatted report"""

    # Convert markdown-style headers to HTML
    html_content = content.replace("## ", "<h2>").replace("\n\n", "</h2>\n")

    report = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        .content {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
    </div>

    <div class="metadata">
        <strong>Type:</strong> {report_type.replace('_', ' ').title()}<br>
        <strong>Date:</strong> {timestamp}<br>
        <strong>Généré par:</strong> RAG Agent System
    </div>

    <div class="content">
        {html_content}
    </div>
</body>
</html>"""

    return report


def _generate_text_report(title: str, content: str, report_type: str, timestamp: str) -> str:
    """Generate a plain text formatted report"""

    report = f"""
{'='*80}
{title.upper()}
{'='*80}

Type: {report_type.replace('_', ' ').title()}
Date: {timestamp}
Généré par: RAG Agent System

{'-'*80}

{content}

{'-'*80}
Fin du rapport
{'='*80}
"""

    return report


# Wrapper function for easy use without decorator
def write_report(title: str, content: str, report_type: str = "general", format: str = "markdown") -> str:
    """
    Write a professional report with proper formatting.
    Uses the global configuration set via set_config().

    Args:
        title: Title of the report
        content: Main content/analysis for the report
        report_type: Type of report (general, technical, business, customer_analysis)
        format: Output format (markdown, text, html)

    Returns:
        Formatted report as a string
    """
    return report_writer_tool.invoke({
        "title": title,
        "content": content,
        "report_type": report_type,
        "format": format
    })
