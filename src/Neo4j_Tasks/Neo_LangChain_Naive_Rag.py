from neo4j import GraphDatabase
import time
from data_manager import DataManager
from rouge_score import rouge_scorer
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class Neo4jDocument(BaseModel):
    page_content: str = Field(..., description="The content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata of the document")

class Neo4jRetriever(BaseRetriever):
    data_manager: Any
    k: int = 2
    
    def _get_relevant_documents(self, query: str) -> List[Neo4jDocument]:
        # Get similar chunks using data manager
        similar_chunks = self.data_manager.get_similar_chunks(query, self.k)
        
        # Convert to LangChain documents
        documents = []
        for chunk in similar_chunks:
            doc = Neo4jDocument(
                page_content=chunk['text'],
                metadata={"score": chunk['score']}
            )
            documents.append(doc)
        
        return documents

class NeoLangChainNaiveRAG:
    def __init__(self, driver=None):
        self.data_manager = DataManager(driver)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize OpenAI embeddings instead of HuggingFace
        # No API key needed as it will use the OPENAI_API_KEY environment variable
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize retriever
        self.retriever = Neo4jRetriever(data_manager=self.data_manager)

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        start_time = time.time()
        
        # Update k value for the retriever
        self.retriever.k = k
        
        # Get similar chunks using LangChain retriever
        documents = self.retriever._get_relevant_documents(query)
        
        results = []
        for doc in documents:
            # Calculate ROUGE scores if ground truth is available
            scores = self.calculate_rouge_scores(doc.page_content, ground_truth) if ground_truth else {
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
            
            results.append({
                'text': doc.page_content,
                'score': doc.metadata.get('score', 0),
                'latency': time.time() - start_time,
                **scores
            })
            
        return results

    def calculate_rouge_scores(self, prediction, ground_truth):
        if not ground_truth:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        
        # Calculate ROUGE scores against all ground truth passages
        scores = []
        for truth in ground_truth:
            score = self.scorer.score(prediction, truth)
            scores.append({
                'precision': score['rouge1'].precision,
                'recall': score['rouge1'].recall,
                'f1': score['rouge1'].fmeasure
            })
        
        # Return the best scores
        best_score = max(scores, key=lambda x: x['f1'])
        return best_score