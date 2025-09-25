"""Embedding and vector store components for RAG"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json


class EmbeddingModel(Enum):
    """Available embedding models"""
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    AZURE_OPENAI = "azure-openai-embedding"
    LOCAL = "local-embedding"


@dataclass
class Document:
    """Document structure for vector store"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class EmbeddingService:
    """Service for creating embeddings"""

    def __init__(self, model: EmbeddingModel, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        """Get embedding dimension for model"""
        dimensions = {
            EmbeddingModel.OPENAI_ADA: 1536,
            EmbeddingModel.OPENAI_3_SMALL: 1536,
            EmbeddingModel.OPENAI_3_LARGE: 3072,
            EmbeddingModel.AZURE_OPENAI: 1536,
            EmbeddingModel.LOCAL: 768
        }
        return dimensions.get(self.model, 768)

    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding for text"""
        # Simulated embedding - in real implementation would call API
        # For demonstration, create random embedding
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for multiple texts"""
        return [self.embed_text(text) for text in texts]


class SimilarityMetric(Enum):
    """Similarity metrics for vector search"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class VectorStore:
    """Simple in-memory vector store"""

    def __init__(self, embedding_service: EmbeddingService, metric: SimilarityMetric = SimilarityMetric.COSINE):
        self.embedding_service = embedding_service
        self.metric = metric
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_service.embed_text(doc.content)
            self.documents.append(doc)

        # Update embeddings matrix
        self._update_embeddings_matrix()

    def _update_embeddings_matrix(self):
        """Update the embeddings matrix for efficient search"""
        if self.documents:
            self.embeddings = np.array([doc.embedding for doc in self.documents])

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if not self.documents:
            return []

        # Embed query
        query_embedding = self.embedding_service.embed_text(query)

        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding)

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Return documents with scores
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(similarities[idx])))

        return results

    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate similarities between query and documents"""
        if self.embeddings is None:
            return np.array([])

        if self.metric == SimilarityMetric.COSINE:
            # Cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            similarities = np.dot(doc_norms, query_norm)

        elif self.metric == SimilarityMetric.EUCLIDEAN:
            # Negative euclidean distance (so higher is better)
            distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
            similarities = -distances

        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product
            similarities = np.dot(self.embeddings, query_embedding)

        else:
            similarities = np.zeros(len(self.embeddings))

        return similarities

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """Hybrid search combining vector and keyword search"""
        # Vector search
        vector_results = self.similarity_search(query, k=k*2)

        # Keyword search (simple BM25-like scoring)
        keyword_scores = self._keyword_search(query)

        # Combine scores
        combined_scores = {}

        for doc, score in vector_results:
            combined_scores[doc.id] = alpha * score

        for doc_id, score in keyword_scores.items():
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * score
            else:
                combined_scores[doc_id] = (1 - alpha) * score

        # Sort and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Convert back to documents
        results = []
        for doc_id, score in sorted_results:
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                results.append((doc, score))

        return results

    def _keyword_search(self, query: str) -> Dict[str, float]:
        """Simple keyword-based search"""
        query_terms = query.lower().split()
        scores = {}

        for doc in self.documents:
            doc_terms = doc.content.lower().split()
            score = 0

            for term in query_terms:
                score += doc_terms.count(term)

            if score > 0:
                # Normalize by document length
                scores[doc.id] = score / len(doc_terms)

        return scores


class RAGPipeline:
    """Complete RAG pipeline"""

    def __init__(self, vector_store: VectorStore, llm: Any, reranker: Optional[Any] = None):
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents"""
        results = self.vector_store.similarity_search(query, k=k)
        return [doc for doc, _ in results]

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """Rerank documents"""
        if not self.reranker:
            return documents[:top_k]

        # Simulated reranking - in real implementation would use reranker model
        # For now, just return top_k documents
        return documents[:top_k]

    def generate(self, query: str, context: List[Document]) -> str:
        """Generate response using retrieved context"""
        # Format context
        context_str = "\n\n".join([
            f"Document {i+1}:\n{doc.content}"
            for i, doc in enumerate(context)
        ])

        # Create prompt
        prompt = f"""Answer the following question based on the provided context.

Context:
{context_str}

Question: {query}

Answer:"""

        # Generate response (simulated)
        response = f"Based on the context, here's the answer to '{query}'"

        return response

    def query(self, query: str, k_retrieve: int = 5, k_rerank: int = 3) -> Dict[str, Any]:
        """Complete RAG query pipeline"""
        # Retrieve
        retrieved_docs = self.retrieve(query, k=k_retrieve)

        # Rerank
        reranked_docs = self.rerank(query, retrieved_docs, top_k=k_rerank)

        # Generate
        response = self.generate(query, reranked_docs)

        return {
            "query": query,
            "retrieved_documents": len(retrieved_docs),
            "reranked_documents": len(reranked_docs),
            "response": response,
            "context": [doc.content[:200] + "..." for doc in reranked_docs]
        }


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing"""
    documents = [
        Document(
            id="doc1",
            content="RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation for better AI responses.",
            metadata={"source": "tutorial", "topic": "rag"}
        ),
        Document(
            id="doc2",
            content="Vector databases store embeddings and enable similarity search for semantic retrieval.",
            metadata={"source": "tutorial", "topic": "embeddings"}
        ),
        Document(
            id="doc3",
            content="LangChain provides tools for building applications with large language models.",
            metadata={"source": "documentation", "topic": "tools"}
        ),
        Document(
            id="doc4",
            content="Agents can use tools to perform complex tasks by breaking them down into smaller steps.",
            metadata={"source": "guide", "topic": "agents"}
        ),
        Document(
            id="doc5",
            content="The Model Context Protocol (MCP) standardizes how AI models interact with external tools.",
            metadata={"source": "specification", "topic": "mcp"}
        )
    ]
    return documents


def demonstrate_embedding_similarity():
    """Demonstrate embedding similarity calculations"""
    service = EmbeddingService(EmbeddingModel.LOCAL)

    texts = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "The dog played in the park",
        "Machine learning is fascinating"
    ]

    embeddings = service.embed_batch(texts)

    print("Similarity Matrix:")
    print("-" * 50)

    for i, text1 in enumerate(texts):
        similarities = []
        for j, text2 in enumerate(texts):
            sim = np.dot(embeddings[i], embeddings[j])
            similarities.append(f"{sim:.2f}")

        print(f"Text {i+1}: {similarities}")

    return embeddings