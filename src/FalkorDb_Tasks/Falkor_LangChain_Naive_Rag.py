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

load_dotenv()

class FalkorLangChainNaiveRAG:
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
        
    def _initialize_vector_store(self):
        """Initialize or load vector store with content from FalkorDB"""
        # Check if we have a persisted vector store
        vector_store_path = "faiss_db/falkor_langchain_naive"
        if os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return
            
        # If not, we need to create one from scratch
        print("Building new vector store for LangChain Naive RAG...")
        
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
                    'topic': article.get('topic', '')
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
        
    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve relevant documents using basic LangChain RAG"""
        start_time = time.time()
        
        # Perform similarity search
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            # Convert distance score to similarity score (closer to 1 is better)
            similarity_score = 1.0 / (1.0 + score)
            
            result = {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': similarity_score,
                'latency': time.time() - start_time
            }
            
            # Calculate metrics if ground truth is provided
            if ground_truth:
                rouge_scores = [self.scorer.score(doc.page_content, gt) for gt in ground_truth]
                if rouge_scores:
                    best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                    result['precision'] = best_score['rougeL'].precision
                    result['recall'] = best_score['rougeL'].recall
                    result['f1'] = best_score['rougeL'].fmeasure
            else:
                result['precision'] = result['recall'] = result['f1'] = 0.0
                
            results.append(result)
            
        return results
        
    def close(self):
        """Only close if we created our own data manager"""
        if not self._shared_data_manager:
            self.data_manager.close()