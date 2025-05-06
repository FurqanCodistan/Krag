from data_manager import PostgresDataManager
from rouge_score import rouge_scorer
import time

class PostgresNaiveRAG:
    def __init__(self, conn=None):
        self.data_manager = PostgresDataManager(conn)

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using basic similarity search"""
        start_time = time.time()
        
        # Get similar chunks using FAISS
        results = self.data_manager.get_similar_chunks(query, k)
        
        # Add latency to results
        latency = time.time() - start_time
        for result in results:
            result['latency'] = latency
        
        # Calculate metrics using provided ground truth
        if ground_truth:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            for result in results:
                rouge_scores = [scorer.score(result["text"], gt) for gt in ground_truth]
                best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                result['precision'] = best_score['rougeL'].precision
                result['recall'] = best_score['rougeL'].recall
                result['f1'] = best_score['rougeL'].fmeasure
        else:
            for result in results:
                result['precision'] = result['recall'] = result['f1'] = 0.0
        
        return results

    def close(self):
        self.data_manager.close()