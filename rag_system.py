"""
RAG (Retrieval-Augmented Generation) System
Provides real-time information retrieval capabilities
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except ImportError:
    print("⚠️  Installing required packages...")
    os.system("pip install -q sentence-transformers faiss-cpu beautifulsoup4 requests")
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

from post_training_config import RAG_CONFIG, KNOWLEDGE_SOURCES


@dataclass
class Document:
    """Represents a retrieved document"""
    content: str
    source: str
    url: Optional[str] = None
    score: float = 0.0
    metadata: Dict = None


class RAGSystem:
    """
    Retrieval-Augmented Generation System
    Combines model knowledge with real-time information retrieval
    """
    
    def __init__(self, config=None):
        self.config = config or RAG_CONFIG
        self.embedding_model = None
        self.index = None
        self.documents = []
        
        print("🔧 Initializing RAG System...")
        self._initialize()
    
    def _initialize(self):
        """Initialize embedding model and vector database"""
        try:
            print(f"📥 Loading embedding model: {self.config['embedding_model']}")
            self.embedding_model = SentenceTransformer(self.config['embedding_model'])
            print("✅ Embedding model loaded")
            
            # Initialize FAISS index
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_dim)
            print(f"✅ FAISS index initialized (dim={embedding_dim})")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    def search_web(self, query: str, max_results: int = 5) -> List[Document]:
        """
        Search the web using DuckDuckGo (privacy-focused)
        """
        documents = []
        
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Extract abstract
            if data.get("Abstract"):
                documents.append(Document(
                    content=data["Abstract"],
                    source="DuckDuckGo",
                    url=data.get("AbstractURL"),
                    score=1.0
                ))
            
            # Extract related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    documents.append(Document(
                        content=topic["Text"],
                        source="DuckDuckGo",
                        url=topic.get("FirstURL"),
                        score=0.8
                    ))
            
            print(f"✅ Found {len(documents)} web results")
            
        except Exception as e:
            print(f"⚠️  Web search failed: {e}")
        
        return documents
    
    def search_wikipedia(self, query: str) -> List[Document]:
        """
        Search Wikipedia for relevant information
        """
        documents = []
        
        try:
            # Wikipedia API
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 3
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            for result in data.get("query", {}).get("search", []):
                # Get page content
                content_params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "pageids": result["pageid"]
                }
                
                content_response = requests.get(url, params=content_params, timeout=10)
                content_data = content_response.json()
                
                page = content_data.get("query", {}).get("pages", {}).get(str(result["pageid"]), {})
                extract = page.get("extract", "")
                
                if extract:
                    documents.append(Document(
                        content=extract,
                        source="Wikipedia",
                        url=f"https://en.wikipedia.org/?curid={result['pageid']}",
                        score=0.9,
                        metadata={"title": result["title"]}
                    ))
            
            print(f"✅ Found {len(documents)} Wikipedia results")
            
        except Exception as e:
            print(f"⚠️  Wikipedia search failed: {e}")
        
        return documents
    
    def search_github(self, query: str, search_type: str = "code") -> List[Document]:
        """
        Search GitHub for code and repositories
        """
        documents = []
        
        try:
            # GitHub Search API (no auth required for basic search)
            if search_type == "code":
                url = "https://api.github.com/search/code"
            else:
                url = "https://api.github.com/search/repositories"
            
            params = {
                "q": query,
                "per_page": 5
            }
            
            headers = {
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get("items", []):
                    if search_type == "code":
                        content = f"File: {item['name']}\nRepo: {item['repository']['full_name']}\nPath: {item['path']}"
                    else:
                        content = f"Repo: {item['full_name']}\nDescription: {item.get('description', 'N/A')}\nStars: {item['stargazers_count']}"
                    
                    documents.append(Document(
                        content=content,
                        source="GitHub",
                        url=item.get("html_url"),
                        score=0.85
                    ))
                
                print(f"✅ Found {len(documents)} GitHub results")
            
        except Exception as e:
            print(f"⚠️  GitHub search failed: {e}")
        
        return documents
    
    def search_stackoverflow(self, query: str) -> List[Document]:
        """
        Search Stack Overflow for programming Q&A
        """
        documents = []
        
        try:
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": "stackoverflow",
                "filter": "withbody"
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            for item in data.get("items", [])[:3]:
                # Clean HTML from body
                soup = BeautifulSoup(item.get("body", ""), "html.parser")
                body_text = soup.get_text()
                
                content = f"Q: {item['title']}\n\n{body_text[:500]}..."
                
                documents.append(Document(
                    content=content,
                    source="Stack Overflow",
                    url=item.get("link"),
                    score=0.9,
                    metadata={
                        "score": item.get("score", 0),
                        "accepted": item.get("is_answered", False)
                    }
                ))
            
            print(f"✅ Found {len(documents)} Stack Overflow results")
            
        except Exception as e:
            print(f"⚠️  Stack Overflow search failed: {e}")
        
        return documents
    
    def retrieve(self, query: str, sources: List[str] = None) -> List[Document]:
        """
        Retrieve relevant documents from multiple sources
        
        Args:
            query: Search query
            sources: List of sources to search (default: all enabled sources)
        
        Returns:
            List of relevant documents
        """
        all_documents = []
        
        if sources is None:
            sources = []
            if KNOWLEDGE_SOURCES["web_search"]["enabled"]:
                sources.append("web")
            if KNOWLEDGE_SOURCES["wikipedia"]["enabled"]:
                sources.append("wikipedia")
            if KNOWLEDGE_SOURCES["github"]["enabled"]:
                sources.append("github")
            if KNOWLEDGE_SOURCES["stackoverflow"]["enabled"]:
                sources.append("stackoverflow")
        
        print(f"\n🔍 Retrieving information for: '{query}'")
        print(f"📚 Sources: {', '.join(sources)}")
        
        # Search each source
        if "web" in sources:
            all_documents.extend(self.search_web(query))
        
        if "wikipedia" in sources:
            all_documents.extend(self.search_wikipedia(query))
        
        if "github" in sources:
            all_documents.extend(self.search_github(query))
        
        if "stackoverflow" in sources:
            all_documents.extend(self.search_stackoverflow(query))
        
        # Sort by score
        all_documents.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k documents
        top_k = self.config.get("top_k", 5)
        return all_documents[:top_k]
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context for the model
        """
        if not documents:
            return ""
        
        context = "**Retrieved Information:**\n\n"
        
        for i, doc in enumerate(documents, 1):
            context += f"**Source {i}: {doc.source}**\n"
            if doc.url:
                context += f"URL: {doc.url}\n"
            context += f"{doc.content}\n\n"
            context += "---\n\n"
        
        return context
    
    def add_to_index(self, documents: List[Document]):
        """
        Add documents to the vector index for future retrieval
        """
        if not documents:
            return
        
        try:
            # Extract text content
            texts = [doc.content for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Add to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store documents
            self.documents.extend(documents)
            
            print(f"✅ Added {len(documents)} documents to index (total: {len(self.documents)})")
            
        except Exception as e:
            print(f"❌ Failed to add documents to index: {e}")
    
    def search_index(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Search the local vector index
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search index
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                min(top_k, self.index.ntotal)
            )
            
            # Get documents
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc.score = float(1.0 / (1.0 + distance))  # Convert distance to similarity
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"❌ Index search failed: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Test query
    query = "How to implement binary search in Python"
    
    # Retrieve information
    documents = rag.retrieve(query)
    
    # Format context
    context = rag.format_context(documents)
    
    print("\n" + "="*80)
    print("RETRIEVED CONTEXT")
    print("="*80)
    print(context)
    
    # Add to index for future use
    rag.add_to_index(documents)
