import os
import pickle
import math
import re
from collections import Counter
import faiss
from sentence_transformers import SentenceTransformer

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        
        tokenized_corpus = []
        total_len = 0
        for doc in corpus:
            tokens = re.findall(r'\w+', doc.lower())
            tokenized_corpus.append(tokens)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
        
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 1
        
        words_df = Counter()
        for tokens in tokenized_corpus:
            self.doc_freqs.append(Counter(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                words_df[token] += 1
                
        for word, df in words_df.items():
            self.idf[word] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            
    def get_scores(self, query):
        query_tokens = re.findall(r'\w+', query.lower())
        scores = [0.0] * self.corpus_size
        for query_token in query_tokens:
            idf = self.idf.get(query_token, 0)
            if idf == 0:
                continue
            for idx, doc_freq in enumerate(self.doc_freqs):
                tf = doc_freq.get(query_token, 0)
                doc_len = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf * (numerator / denominator)
        return scores

# Extensive set of agricultural subjects (crops, fruits, vegetables, pests, diseases)
AGRI_SUBJECTS = {
    # Cereals & Grains
    'wheat', 'paddy', 'rice', 'maize', 'corn', 'barley', 'mustard', 'chickpea', 'gram', 'lentil', 'pea', 'pigeonpea',
    'arhar', 'tur', 'soybean', 'groundnut', 'sugarcane', 'cotton', 'jute', 'bajra', 'jowar', 'sorghum', 'ragi', 'millet',
    # Vegetables
    'potato', 'tomato', 'onion', 'garlic', 'ginger', 'turmeric', 'chilli', 'brinjal', 'eggplant', 'cabbage', 'cauliflower',
    'broccoli', 'okra', 'bhendi', 'spinach', 'methi', 'coriander', 'carrot', 'radish', 'turnip', 'cucumber', 'gourd',
    # Fruits
    'mango', 'banana', 'citrus', 'orange', 'lemon', 'lime', 'grape', 'guava', 'papaya', 'pomegranate', 'apple', 'pear',
    'peach', 'plum', 'cherry', 'strawberry', 'coconut', 'arecanut', 'cashew',
    # Pests & Diseases
    'aphid', 'aphids', 'jassid', 'jassids', 'thrip', 'thrips', 'whitefly', 'whiteflies', 'bollworm', 'caterpillar',
    'borer', 'rust', 'blight', 'mildew', 'rot', 'wilt', 'virus', 'fungus', 'nematode', 'mite', 'mites', 'insect', 'pests'
}

AGRI_STOPWORDS = {
    'how', 'to', 'what', 'when', 'where', 'why', 'who', 'which', 'should', 'i', 'you', 'he', 'she', 'they',
    'sow', 'plant', 'grow', 'harvest', 'irrigate', 'irrigation', 'water', 'fertilize', 'fertilizer', 'dose',
    'pesticide', 'insecticide', 'fungicide', 'herbicide', 'spray', 'treatment', 'prevent', 'prevention',
    'disease', 'pest', 'insect', 'weed', 'management', 'care', 'control', 'sowing', 'growing', 'harvesting',
    'the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'by', 'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'can', 'do', 'does', 'did', 'use', 'using', 'used', 'best', 'good', 'proper', 'recommend', 'recommended',
    'about', 'any', 'some', 'many', 'much', 'time', 'timing', 'season', 'stage', 'days', 'das', 'month', 'week',
    'crop', 'crops', 'plant', 'plants', 'fruit', 'fruits', 'tree', 'trees', 'cultivation', 'yield'
}

class HybridSearcher:
    def __init__(self, index_dir="faiss_index", model_name="all-MiniLM-L6-v2"):
        print("Loading FAISS index...")
        self.faiss_idx = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "index.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        print("Initializing BM25 on chunks...")
        self.bm25 = BM25(self.chunks)
        
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer(model_name)

    def search(self, query, k=3, alpha=0.4):
        # A. Semantic Search Scores
        q_vec = self.model.encode([query], convert_to_numpy=True)[0]
        candidate_k = min(100, len(self.chunks))
        D, I = self.faiss_idx.search(q_vec.reshape(1, -1), candidate_k)
        
        semantic_scores = [0.0] * len(self.chunks)
        max_sim = 0
        for dist, idx in zip(D[0], I[0]):
            sim = 1.0 / (1.0 + dist)
            semantic_scores[idx] = sim
            if sim > max_sim:
                max_sim = sim
                
        if max_sim > 0:
            semantic_scores = [s / max_sim for s in semantic_scores]
            
        # B. BM25 Scores
        bm25_scores = self.bm25.get_scores(query)
        max_bm25 = max(bm25_scores) if bm25_scores else 0
        if max_bm25 > 0:
            bm25_scores = [s / max_bm25 for s in bm25_scores]
            
        # C. Dynamic Subject Detection
        query_lower = query.lower()
        query_subjects = [s for s in AGRI_SUBJECTS if s in query_lower]
        if "rice" in query_subjects and "paddy" not in query_subjects:
            query_subjects.append("paddy")
        if "paddy" in query_subjects and "rice" not in query_subjects:
            query_subjects.append("rice")
            
        # Fallback: dynamically extract key noun words if no predefined subjects match
        if not query_subjects:
            query_subjects = [w for w in re.findall(r'\w+', query_lower) if w not in AGRI_STOPWORDS and len(w) > 2]

        # D. Combine Scores
        combined_scores = []
        for idx in range(len(self.chunks)):
            sem = semantic_scores[idx]
            keyword = bm25_scores[idx]
            score = alpha * sem + (1.0 - alpha) * keyword
            
            # Apply dynamic subject boosting
            if query_subjects:
                chunk_lower = self.chunks[idx].lower()
                has_target = any(s in chunk_lower for s in query_subjects)
                other_subjects = [s for s in AGRI_SUBJECTS if s not in query_subjects]
                has_other_only = any(s in chunk_lower for s in other_subjects) and not has_target
                
                if has_target:
                    score *= 1.6
                elif has_other_only:
                    score *= 0.4
                    
            combined_scores.append((score, idx))
            
        # Sort and return top k
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = combined_scores[:k]
        
        return [self.chunks[idx] for score, idx in top_k]
