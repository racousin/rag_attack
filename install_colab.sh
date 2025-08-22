#!/bin/bash

echo "Installing RAG Attack dependencies for Google Colab..."

# Update package list
apt-get update -qq

# Install ODBC dependencies
echo "Installing ODBC drivers..."
apt-get install -y -qq unixodbc unixodbc-dev

# Install MS ODBC Driver 18
curl -s https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
apt-get install -y -qq curl gnupg
curl -s https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list
apt-get update -qq
ACCEPT_EULA=Y apt-get install -y -qq msodbcsql18

# Install Python dependencies
echo "Installing Python packages..."
pip install -q \
    azure-core>=1.35.0 \
    azure-identity>=1.23.0 \
    azure-search==1.0.0b2 \
    azure-search-documents>=11.5.3 \
    certifi>=2025.6.15 \
    python-dotenv>=0.9.9 \
    faiss-cpu>=1.11.0 \
    faker>=23.4.0 \
    fastapi>=0.116.1 \
    ipykernel>=6.29.5 \
    langchain-community>=0.3.27 \
    langchain-core>=0.3.67 \
    langchain-huggingface>=0.3.0 \
    langchain-openai>=0.3.27 \
    langgraph>=0.5.1 \
    matplotlib>=3.10.3 \
    numpy>=2.3.1 \
    pandas>=2.3.0 \
    pyodbc>=5.2.0 \
    requests>=2.32.4 \
    seaborn>=0.13.2 \
    sentence-transformers>=5.0.0 \
    streamlit>=1.47.0 \
    uvicorn>=0.35.0

# Install Azure management libraries
pip install -q \
    azure-mgmt-resource \
    azure-mgmt-search \
    azure-mgmt-sql \
    azure-mgmt-cognitiveservices \
    azure-mgmt-web \
    azure-keyvault-secrets

echo "Installation complete!"