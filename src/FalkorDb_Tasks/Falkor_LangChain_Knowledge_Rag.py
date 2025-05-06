from data_manager import FalkorDataManager
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rouge_score import rouge_scorer
import time
import os
from dotenv import load_dotenv
import torch
import networkx as nx
from typing import List, Dict, Any

load_dotenv()

class FalkorLangChainKnowledgeRAG:
    def __init__(self, data_manager=None, embedding_model_name="text-embedding-ada-002"):
        """Initialize with existing data manager or create new one"""
        if isinstance(data_manager, FalkorDataManager):
            self.data_manager = data_manager
            self._shared_data_manager = True
        else:
            # Create a new data manager if none was provided
            from falkordb import FalkorDB
            host = "localhost"
            port = 6379
            db = FalkorDB(host=host, port=port)
            self.data_manager = FalkorDataManager(db)
            self._shared_data_manager = False
        
        # Initialize LangChain embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            # openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Initialize ROUGE scorer
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Build knowledge graph
        self.graph = nx.DiGraph()
        self._build_knowledge_graph()
        
    def _initialize_vector_store(self):
        """Initialize or load vector store with content from FalkorDB"""
        # Check if we have a persisted vector store
        vector_store_path = "faiss_db/falkor_langchain_knowledge"
        if os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return
            
        # If not, we need to create one from scratch
        print("Building new vector store for LangChain Knowledge RAG...")
        
        # Get all articles from FalkorDB
        articles = self.data_manager.get_all_articles()
        
        if not articles:
            raise ValueError("No articles found in FalkorDB")
            
        # Process articles for vector store
        texts = []
        metadatas = []
        
        for article in articles:
            # Split content into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(article['content'])
            
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    'article_id': article['article_id'],
                    'title': article['title'],
                    'topic': article.get('topic', ''),
                    'chunk_index': len(texts) - 1  # Track the chunk index
                })
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts, 
            self.embeddings, 
            metadatas=metadatas
        )
        
        # Persist vector store
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        self.vector_store.save_local(vector_store_path)
    
    def _build_knowledge_graph(self):
        """Build a knowledge graph using data from FalkorDB"""
        print("Building knowledge graph for LangChain Knowledge RAG...")
        
        # Get article data
        articles = self.data_manager.get_all_articles()
        authors = self.data_manager.get_all_authors()
        categories = self.data_manager.get_all_categories()
        article_authors = self.data_manager.get_article_authors()
        article_categories = self.data_manager.get_article_categories()
        
        # Add article nodes
        for article in articles:
            self.graph.add_node(
                f"article_{article['article_id']}", 
                type='article',
                title=article['title'],
                content=article['content'], 
                topic=article.get('topic', '')
            )
        
        # Add author nodes and edges
        for author in authors:
            self.graph.add_node(
                f"author_{author['author_id']}", 
                type='author', 
                name=author['name']
            )
        
        for aa in article_authors:
            self.graph.add_edge(
                f"article_{aa['article_id']}", 
                f"author_{aa['author_id']}", 
                type='written_by'
            )
        
        # Add category nodes and edges
        for category in categories:
            self.graph.add_node(
                f"category_{category['category_id']}", 
                type='category', 
                name=category['name']
            )
        
        for ac in article_categories:
            self.graph.add_edge(
                f"article_{ac['article_id']}", 
                f"category_{ac['category_id']}", 
                type='belongs_to',
                is_primary=ac.get('is_primary', False)
            )
        
        # Add article relationships
        relationships = self.data_manager.get_article_relationships()
        for rel in relationships:
            self.graph.add_edge(
                f"article_{rel['source_article_id']}", 
                f"article_{rel['target_article_id']}", 
                type=rel['relationship_type'],
                weight=rel.get('strength', 1.0)
            )
    
    def get_article_metadata(self, article_id: str) -> Dict:
        """Get metadata for an article from the knowledge graph"""
        node_id = f"article_{article_id}"
        
        if not self.graph.has_node(node_id):
            return {}
        
        # Get base metadata
        metadata = dict(self.graph.nodes[node_id])
        
        # Add authors
        authors = []
        for _, author_id, data in self.graph.out_edges(node_id, data=True):
            if data.get('type') == 'written_by' and self.graph.nodes[author_id]['type'] == 'author':
                authors.append(self.graph.nodes[author_id]['name'])
        metadata['authors'] = authors
        
        # Add categories
        categories = []
        for _, category_id, data in self.graph.out_edges(node_id, data=True):
            if data.get('type') == 'belongs_to' and self.graph.nodes[category_id]['type'] == 'category':
                categories.append(self.graph.nodes[category_id]['name'])
        metadata['categories'] = categories
        
        return metadata
        
    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using knowledge graph-enhanced retrieval"""
        start_time = time.time()
        
        # First perform basic vector similarity search
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k*2)  # Get more candidates to rerank
        
        # Prepare for reranking
        candidates = []
        for doc, score in docs_with_scores:
            # Extract article ID from metadata
            article_id = doc.metadata.get('article_id')
            if not article_id:
                continue
                
            # Base similarity score (from vector search)
            base_score = 1.0 / (1.0 + score)
            
            # Get article metadata from knowledge graph
            metadata = self.get_article_metadata(article_id)
            
            # Calculate metadata relevance score
            metadata_score = 0.0
            query_terms = set(query.lower().split())
            
            # Match categories
            for category in metadata.get('categories', []):
                if any(term in category.lower() for term in query_terms):
                    metadata_score += 0.15
            
            # Match authors
            for author in metadata.get('authors', []):
                if any(term in author.lower() for term in query_terms):
                    metadata_score += 0.05
            
            # Match topic
            topic = metadata.get('topic', '')
            if topic and any(term in topic.lower() for term in query_terms):
                metadata_score += 0.2
                
            # Combined score with knowledge graph enhancement
            final_score = (0.7 * base_score) + (0.3 * metadata_score)
            
            candidates.append({
                'text': doc.page_content,
                'article_id': article_id,
                'metadata': metadata,
                'base_score': base_score,
                'metadata_score': metadata_score,
                'score': final_score,
                'latency': time.time() - start_time
            })
        
        # Sort by enhanced score and take top k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        results = candidates[:k]
        
        # Calculate metrics if ground truth is provided
        if ground_truth:
            for result in results:
                rouge_scores = [self.scorer.score(result['text'], gt) for gt in ground_truth]
                if rouge_scores:
                    best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                    result['precision'] = best_score['rougeL'].precision
                    result['recall'] = best_score['rougeL'].recall
                    result['f1'] = best_score['rougeL'].fmeasure
                else:
                    result['precision'] = result['recall'] = result['f1'] = 0.0
        else:
            for result in results:
                result['precision'] = result['recall'] = result['f1'] = 0.0
                
        return results
        
    def close(self):
        """Only close if we created our own data manager"""
        if not self._shared_data_manager:
            self.data_manager.close()