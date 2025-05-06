from neo4j import GraphDatabase
import time
from data_manager import DataManager
from rouge_score import rouge_scorer

class NeoNaiveRAG:
    def __init__(self, driver=None):
        self.data_manager = DataManager(driver)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        start_time = time.time()
        
        # Get similar chunks using basic similarity search
        similar_chunks = self.data_manager.get_similar_chunks(query, k)
        
        results = []
        for chunk in similar_chunks:
            # Calculate ROUGE scores if ground truth is available
            scores = self.calculate_rouge_scores(chunk['text'], ground_truth) if ground_truth else {
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
            
            results.append({
                'text': chunk['text'],
                'score': chunk['score'],
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
