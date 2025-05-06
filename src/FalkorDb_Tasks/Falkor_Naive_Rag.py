from data_manager import FalkorDataManager
from rouge_score import rouge_scorer
import time
import torch

class FalkorNaiveRAG:
    def __init__(self, data_manager=None):
        """Initialize with existing data manager or create new one"""
        if isinstance(data_manager, FalkorDataManager):
            self.data_manager = data_manager
        else:
            self.data_manager = FalkorDataManager(data_manager)  # data_manager here would be the connection

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using basic similarity search"""
        start_time = time.time()
        
        # Get similar content using cosine similarity
        results = self.data_manager.get_similar_content(query, k)
        
        # Add latency and metrics
        enhanced_results = []
        for result in results:
            result['latency'] = time.time() - start_time
            
            # Calculate metrics if ground truth is provided
            if ground_truth:
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                rouge_scores = [scorer.score(result["text"], gt) for gt in ground_truth]
                best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                result['precision'] = best_score['rougeL'].precision
                result['recall'] = best_score['rougeL'].recall
                result['f1'] = best_score['rougeL'].fmeasure
            else:
                result['precision'] = result['recall'] = result['f1'] = 0.0
                
            enhanced_results.append(result)
        
        return enhanced_results

    def close(self):
        """Only close if we created our own data manager"""
        if not hasattr(self, '_shared_data_manager'):
            self.data_manager.close()