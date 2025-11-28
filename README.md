# RAG Attack - Formation LLM & Agents

Formation pratique sur les systèmes RAG (Retrieval-Augmented Generation) et les agents LLM.

**Cas d'usage** : VéloCorp, entreprise fictive de vente de vélos.

---

## Programme

### Échauffement

| Notebook | Description | Lien |
|----------|-------------|------|
| **Warm-up Jupyter** | Prise en main de Jupyter et Python | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/rag_attack/blob/main/warm_up_jupyter_notebook.ipynb) |

---

### Matinée : RAG (Retrieval-Augmented Generation)

Construire un système RAG sur les données VéloCorp.

| Notebook | Description | Lien |
|----------|-------------|------|
| **Partie 0 - Données** | Présentation des données | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/rag_attack/blob/main/partie1_rag_0.ipynb) |
| **Partie 1 - RAG** | Comprendre et implémenter un RAG : embeddings, recherche vectorielle, génération | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/rag_attack/blob/main/partie1_rag_1.ipynb) |

**Concepts abordés :**
- Embeddings et recherche sémantique
- Azure Cognitive Search (keyword, vector, hybrid)
- Prompt engineering pour RAG
- Évaluation de la qualité des réponses

---

### Après-midi : Agents LLM

Utiliser des agents sur les données VéloCorp.

| Notebook | Description | Lien |
|----------|-------------|------|
| **Partie 2.1 - Outils & LangGraph** | Comprendre les outils (function calling) et introduction à LangGraph | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/rag_attack/blob/main/partie2_agentic_1.ipynb) |
| **Partie 2.2 - Agents RAG** | Utiliser SimpleAgent et ReflectionAgent avec les 6 outils VéloCorp | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/rag_attack/blob/main/partie2_agentic_2.ipynb) |

**Concepts abordés :**
- Function calling et schémas d'outils
- LangGraph : graphes d'états pour agents
- Outils multi-sources : CRM, ERP, RAG, Web, Email, Writer
- Patterns d'agents : Simple vs Reflection

---

## Installation

```bash
pip install git+https://github.com/racousin/rag_attack.git
```

## Configuration

```python
from rag_attack import set_config

config = {
    "openai_endpoint": "...",
    "openai_key": "...",
    "chat_deployment": "...",
    "search_endpoint": "...",
    "search_key": "...",
    # ...
}

set_config(config)
```

---

## Structure du projet

```
rag_attack/
├── warm_up_jupyter_notebook.ipynb  # Échauffement
├── partie1_rag_1.ipynb             # RAG
├── partie2_agentic_1.ipynb         # Outils & LangGraph
├── partie2_agentic_2.ipynb         # Agents RAG
└── rag_attack/                     # Package Python
    ├── agents/                     # SimpleAgent, ReflectionAgent
    └── tools/                      # 6 outils (CRM, ERP, RAG, etc.)
```
