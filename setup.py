from setuptools import setup, find_packages

setup(
    name="rag_attack",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "azure-core>=1.35.0",
        "azure-identity>=1.23.0",
        "azure-search-documents>=11.5.3",
        "langchain-community>=0.3.27",
        "langchain-openai>=0.3.27",
        "langgraph>=0.5.1",
        "pyodbc>=5.2.0",
        "requests>=2.32.4",
    ],
)