"""
ChromaDB Manager for OLIST Logistics RAG System
Enterprise-grade vector database management with Ollama embeddings
"""
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import ollama
from typing import List, Dict, Optional
from logistics_knowledge_base import LOGISTICS_DOCUMENTS, create_knowledge_base


class ChromaDBManager:
    """
    Manages ChromaDB operations for logistics knowledge base
    Uses Ollama mxbai-embed-large for embeddings (1024-dimensional)
    """
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        collection_name: str = "olist_logistics_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB with persistent storage
        
        Args:
            persist_directory: Path for storing vector database
            collection_name: Name of the collection
            embedding_model: Sentence transformer model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name
            )
            print(f"✓ Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection with sentence transformers
            print(f"✓ Creating new collection: {self.collection_name}")
            
            # Use sentence transformers (lightweight, no Ollama needed)
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
                metadata={"description": "OLIST Logistics Knowledge Base"}
            )
            return collection
    
    def index_knowledge_base(self, force_reindex: bool = False):
        """
        Index all documents from knowledge base into ChromaDB
        
        Args:
            force_reindex: If True, clear and reindex all documents
        """
        if force_reindex:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
        
        # Check if already indexed
        if self.collection.count() > 0 and not force_reindex:
            print(f"✓ Collection already indexed ({self.collection.count()} documents)")
            return
        
        print("📚 Indexing knowledge base into ChromaDB...")
        
        # Create knowledge base files
        create_knowledge_base()
        
        documents = []
        metadatas = []
        ids = []
        
        doc_id = 0
        for filename, content in LOGISTICS_DOCUMENTS.items():
            # Split content into chunks (by sections)
            sections = self._split_document(content, filename)
            
            for i, section in enumerate(sections):
                if section.strip():
                    documents.append(section)
                    metadatas.append({
                        "source": filename,
                        "section_id": i,
                        "category": self._categorize_document(filename)
                    })
                    ids.append(f"{filename}_{doc_id}")
                    doc_id += 1
        
        # Add documents to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ Indexed {len(documents)} document chunks")
        else:
            print("⚠ No documents to index")
    
    def _split_document(self, content: str, filename: str) -> List[str]:
        """
        Split document into semantic chunks
        
        Args:
            content: Document content
            filename: Document filename for context
            
        Returns:
            List of document chunks
        """
        # Split by major sections (double newline)
        chunks = []
        current_chunk = []
        
        for line in content.split('\n'):
            if line.strip():
                current_chunk.append(line)
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _categorize_document(self, filename: str) -> str:
        """Categorize document by filename"""
        categories = {
            "carrier_rules.txt": "carrier_optimization",
            "distance_guidelines.txt": "delivery_time",
            "weekend_holidays.txt": "temporal_factors",
            "weight_volume_rules.txt": "package_characteristics",
            "customer_recovery.txt": "customer_retention",
            "payment_lag_impact.txt": "payment_risk"
        }
        return categories.get(filename, "general")
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        filter_category: Optional[str] = None
    ) -> Dict:
        """
        Query the knowledge base
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            filter_category: Optional category filter
            
        Returns:
            Query results with documents and metadata
        """
        where_filter = {"category": filter_category} if filter_category else None
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        
        return {
            "documents": results['documents'][0] if results['documents'] else [],
            "metadatas": results['metadatas'][0] if results['metadatas'] else [],
            "distances": results['distances'][0] if results['distances'] else []
        }
    
    def get_relevant_context(
        self,
        predicted_days: float,
        promised_days: float,
        input_data: Dict
    ) -> str:
        """
        Get relevant context for agent based on scenario
        
        Args:
            predicted_days: Predicted delivery time
            promised_days: Promised delivery time
            input_data: Input data dictionary
            
        Returns:
            Formatted context string for agent
        """
        distance = input_data['distance_km']
        weight = input_data['product_weight_g']
        payment_lag = input_data['payment_lag_days']
        is_weekend = input_data['is_weekend_order']
        delay_risk = predicted_days - promised_days
        
        # Build contextual queries
        queries = []
        
        # Distance-based query
        if distance <= 100:
            queries.append("local delivery under 100km guidelines")
        elif distance <= 500:
            queries.append("regional delivery 100-500km recommendations")
        elif distance <= 1500:
            queries.append("interstate delivery 500-1500km guidelines")
        else:
            queries.append("long-distance delivery over 1500km critical guidelines")
        
        # Weight-based query
        if weight < 500:
            queries.append("lightweight packages under 500g")
        elif weight < 2000:
            queries.append("medium packages 500g-2kg")
        elif weight < 5000:
            queries.append("heavy packages 2-5kg")
        else:
            queries.append("very heavy packages over 5kg")
        
        # Temporal factors
        if is_weekend == 1:
            queries.append("weekend delivery penalty impact")
        
        # Payment lag
        if payment_lag > 3:
            queries.append("delayed payment impact on delivery")
        
        # Delay risk
        if delay_risk > 1:
            queries.append("carrier optimization rules for delays")
            queries.append("customer recovery voucher strategy")
        
        # Query ChromaDB for each context
        context_parts = []
        for query in queries:
            results = self.query(query, n_results=2)
            for doc in results['documents']:
                context_parts.append(doc)
        
        return "\n\n---\n\n".join(context_parts[:8])  # Limit to top 8 chunks
    
    def reset_database(self):
        """Reset the entire database (use with caution)"""
        self.client.delete_collection(name=self.collection_name)
        print("✓ Database reset complete")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model
        }


# Test functionality
if __name__ == "__main__":
    print("🔧 Initializing ChromaDB Manager...")
    
    manager = ChromaDBManager()
    
    print("\n📊 Indexing knowledge base...")
    manager.index_knowledge_base(force_reindex=False)
    
    print("\n📈 Database Stats:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 Testing query...")
    test_query = "What carrier should I use for heavy packages over 5kg?"
    results = manager.query(test_query, n_results=3)
    
    print(f"\nQuery: {test_query}")
    print(f"Results found: {len(results['documents'])}")
    for i, doc in enumerate(results['documents'][:2]):
        print(f"\n--- Result {i+1} ---")
        print(doc[:200] + "...")
    
    print("\n✓ ChromaDB Manager test complete!")
