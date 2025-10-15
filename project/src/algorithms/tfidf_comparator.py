"""
Implementação do algoritmo TF-IDF com similaridade de cosseno.
Baseado na especificação do TCC para análise lexical de textos.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from src.utils.text_preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFComparator:
    """
    Implementa comparação de textos usando TF-IDF e similaridade de cosseno.
    
    O algoritmo TF-IDF (Term Frequency-Inverse Document Frequency) combina:
    - TF: Frequência do termo no documento
    - IDF: Frequência inversa do documento no corpus
    
    A similaridade é calculada usando o cosseno do ângulo entre vetores.
    """
    
    def __init__(self, 
                 language: str = 'portuguese',
                 max_features: Optional[int] = None,
                 min_df: int = 1,
                 max_df: float = 1.0,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_custom_preprocessing: bool = True):
        """
        Inicializa o comparador TF-IDF.
        
        Args:
            language: Idioma para pré-processamento
            max_features: Número máximo de features
            min_df: Frequência mínima do documento
            max_df: Frequência máxima do documento
            ngram_range: Faixa de n-gramas
            use_custom_preprocessing: Usar pré-processamento customizado
        """
        self.language = language
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_custom_preprocessing = use_custom_preprocessing
        
        # Inicializa pré-processador
        self.preprocessor = TextPreprocessor(language=language)
        
        # Inicializa vetorizador TF-IDF
        self.vectorizer = None
        self.feature_names = None
        self.corpus_vectors = None
        self.corpus_texts = []
        
        # Métricas de comparação
        self.comparison_results = []
        
    def _preprocess_text(self, text: str) -> str:
        """
        Aplica pré-processamento ao texto.
        
        Args:
            text: Texto a ser processado
            
        Returns:
            str: Texto pré-processado
        """
        if self.use_custom_preprocessing:
            return self.preprocessor.preprocess_for_tfidf(text)
        return text.lower()
    
    def _create_vectorizer(self) -> TfidfVectorizer:
        """
        Cria o vetorizador TF-IDF com configurações específicas.
        
        Returns:
            TfidfVectorizer: Vetorizador configurado
        """
        # Stopwords baseadas no idioma
        if self.language == 'portuguese':
            stop_words = list(self.preprocessor.stopwords)
        elif self.language == 'english':
            stop_words = 'english'
        else:
            stop_words = None
        
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=stop_words,
            ngram_range=self.ngram_range,
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-ZÀ-ÿ]{2,}\b',  # Inclui acentos
            sublinear_tf=True,  # Aplica escala logarítmica ao TF
            norm='l2'  # Normalização L2
        )
        
        return vectorizer
    
    def fit(self, texts: List[str]) -> 'TFIDFComparator':
        """
        Treina o modelo TF-IDF com um corpus de textos.
        
        Args:
            texts: Lista de textos para treinamento
            
        Returns:
            TFIDFComparator: Self para chaining
        """
        if not texts or len(texts) == 0:
            raise ValueError("Lista de textos não pode estar vazia")
        
        logger.info(f"Treinando TF-IDF com {len(texts)} documentos")
        
        # Pré-processa textos
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Cria e treina vetorizador
        self.vectorizer = self._create_vectorizer()
        self.corpus_vectors = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.corpus_texts = texts.copy()
        
        logger.info(f"Vocabulário criado com {len(self.feature_names)} termos")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforma textos em vetores TF-IDF.
        
        Args:
            texts: Lista de textos para transformar
            
        Returns:
            np.ndarray: Matriz de vetores TF-IDF
        """
        if self.vectorizer is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        processed_texts = [self._preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """
        Compara dois textos usando TF-IDF e similaridade de cosseno.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Dict: Resultados da comparação
        """
        if not text1 or not text2:
            raise ValueError("Textos não podem estar vazios")
        
        # Se não há corpus treinado, treina com os dois textos
        if self.vectorizer is None:
            self.fit([text1, text2])
            vectors = self.corpus_vectors
        else:
            # Transforma os textos usando o modelo existente
            vectors = self.transform([text1, text2])
        
        # Calcula similaridade de cosseno
        similarity_matrix = cosine_similarity(vectors)
        similarity_score = similarity_matrix[0, 1]
        
        # Análise detalhada
        analysis = self._detailed_analysis(text1, text2, vectors)
        
        result = {
            'similarity_score': float(similarity_score),
            'similarity_percentage': float(similarity_score * 100),
            'algorithm': 'TF-IDF + Cosine Similarity',
            'text1_length': len(text1),
            'text2_length': len(text2),
            'analysis': analysis,
            'interpretation': self._interpret_similarity(similarity_score)
        }
        
        # Armazena resultado
        self.comparison_results.append(result)
        
        return result
    
    def _detailed_analysis(self, text1: str, text2: str, vectors) -> Dict:
        """
        Realiza análise detalhada da comparação.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            vectors: Vetores TF-IDF
            
        Returns:
            Dict: Análise detalhada
        """
        # Estatísticas dos textos
        stats1 = self.preprocessor.get_text_statistics(text1)
        stats2 = self.preprocessor.get_text_statistics(text2)
        
        # Termos mais importantes de cada texto
        vector1 = vectors[0].toarray().flatten()
        vector2 = vectors[1].toarray().flatten()
        
        # Top termos por TF-IDF score
        top_terms1 = self._get_top_terms(vector1, n=10)
        top_terms2 = self._get_top_terms(vector2, n=10)
        
        # Termos em comum
        common_terms = self._get_common_terms(vector1, vector2, n=10)
        
        # Diversidade lexical
        lexical_diversity = self._calculate_lexical_diversity(text1, text2)
        
        return {
            'text1_stats': stats1,
            'text2_stats': stats2,
            'top_terms_text1': top_terms1,
            'top_terms_text2': top_terms2,
            'common_terms': common_terms,
            'lexical_diversity': lexical_diversity,
            'vocabulary_overlap': self._calculate_vocabulary_overlap(text1, text2)
        }
    
    def _get_top_terms(self, vector: np.ndarray, n: int = 10) -> List[Tuple[str, float]]:
        """
        Obtém os termos com maior score TF-IDF.
        
        Args:
            vector: Vetor TF-IDF
            n: Número de termos a retornar
            
        Returns:
            List[Tuple[str, float]]: Lista de (termo, score)
        """
        if self.feature_names is None:
            return []
        
        # Obtém índices dos maiores valores
        top_indices = np.argsort(vector)[-n:][::-1]
        
        return [(self.feature_names[i], float(vector[i])) 
                for i in top_indices if vector[i] > 0]
    
    def _get_common_terms(self, vector1: np.ndarray, vector2: np.ndarray, 
                         n: int = 10) -> List[Tuple[str, float, float]]:
        """
        Obtém termos comuns entre dois vetores.
        
        Args:
            vector1: Primeiro vetor
            vector2: Segundo vetor
            n: Número de termos a retornar
            
        Returns:
            List[Tuple[str, float, float]]: Lista de (termo, score1, score2)
        """
        if self.feature_names is None:
            return []
        
        # Encontra índices onde ambos vetores têm valores > 0
        common_indices = np.where((vector1 > 0) & (vector2 > 0))[0]
        
        if len(common_indices) == 0:
            return []
        
        # Calcula score combinado (média dos scores)
        combined_scores = (vector1[common_indices] + vector2[common_indices]) / 2
        
        # Ordena por score combinado
        sorted_indices = np.argsort(combined_scores)[-n:][::-1]
        
        result = []
        for idx in sorted_indices:
            original_idx = common_indices[idx]
            term = self.feature_names[original_idx]
            score1 = float(vector1[original_idx])
            score2 = float(vector2[original_idx])
            result.append((term, score1, score2))
        
        return result
    
    def _calculate_lexical_diversity(self, text1: str, text2: str) -> Dict:
        """
        Calcula diversidade lexical entre textos.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Dict: Métricas de diversidade lexical
        """
        tokens1 = set(self.preprocessor.preprocess_text(text1))
        tokens2 = set(self.preprocessor.preprocess_text(text2))
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        return {
            'unique_tokens_text1': len(tokens1),
            'unique_tokens_text2': len(tokens2),
            'common_tokens': len(intersection),
            'total_unique_tokens': len(union),
            'jaccard_similarity': jaccard_similarity
        }
    
    def _calculate_vocabulary_overlap(self, text1: str, text2: str) -> float:
        """
        Calcula sobreposição de vocabulário.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            float: Percentual de sobreposição
        """
        words1 = set(self.preprocessor.preprocess_text(text1))
        words2 = set(self.preprocessor.preprocess_text(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        smaller_set = min(len(words1), len(words2))
        
        return len(intersection) / smaller_set if smaller_set > 0 else 0.0
    
    def _interpret_similarity(self, score: float) -> str:
        """
        Interpreta o score de similaridade.
        
        Args:
            score: Score de similaridade (0-1)
            
        Returns:
            str: Interpretação textual
        """
        if score >= 0.9:
            return "Textos muito similares (quase idênticos)"
        elif score >= 0.7:
            return "Textos similares (alta correspondência)"
        elif score >= 0.5:
            return "Textos moderadamente similares"
        elif score >= 0.3:
            return "Textos com baixa similaridade"
        else:
            return "Textos muito diferentes"
    
    def batch_compare(self, texts: List[str]) -> pd.DataFrame:
        """
        Compara múltiplos textos em lote.
        
        Args:
            texts: Lista de textos para comparar
            
        Returns:
            pd.DataFrame: Matriz de similaridades
        """
        if len(texts) < 2:
            raise ValueError("Necessário pelo menos 2 textos para comparação")
        
        # Treina modelo se necessário
        if self.vectorizer is None:
            self.fit(texts)
        else:
            vectors = self.transform(texts)
        
        # Calcula matriz de similaridade
        similarity_matrix = cosine_similarity(vectors)
        
        # Cria DataFrame
        df = pd.DataFrame(
            similarity_matrix,
            index=[f"Texto_{i+1}" for i in range(len(texts))],
            columns=[f"Texto_{i+1}" for i in range(len(texts))]
        )
        
        return df
    
    def get_feature_importance(self, text: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Obtém importância das features para um texto.
        
        Args:
            text: Texto para análise
            top_n: Número de features a retornar
            
        Returns:
            List[Tuple[str, float]]: Lista de (feature, importância)
        """
        if self.vectorizer is None:
            raise ValueError("Modelo não foi treinado")
        
        vector = self.transform([text])
        return self._get_top_terms(vector.toarray().flatten(), top_n)
    
    def save_model(self, filepath: str):
        """
        Salva o modelo treinado.
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        import pickle
        
        model_data = {
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'corpus_vectors': self.corpus_vectors,
            'corpus_texts': self.corpus_texts,
            'config': {
                'language': self.language,
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo salvo em {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carrega modelo salvo.
        
        Args:
            filepath: Caminho do modelo salvo
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.feature_names = model_data['feature_names']
        self.corpus_vectors = model_data['corpus_vectors']
        self.corpus_texts = model_data['corpus_texts']
        
        # Restaura configurações
        config = model_data['config']
        self.language = config['language']
        self.max_features = config['max_features']
        self.min_df = config['min_df']
        self.max_df = config['max_df']
        self.ngram_range = config['ngram_range']
        
        logger.info(f"Modelo carregado de {filepath}")
    
    def get_statistics(self) -> Dict:
        """
        Obtém estatísticas do modelo.
        
        Returns:
            Dict: Estatísticas do modelo
        """
        if self.vectorizer is None:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'vocabulary_size': len(self.feature_names),
            'corpus_size': len(self.corpus_texts),
            'comparisons_made': len(self.comparison_results),
            'ngram_range': self.ngram_range,
            'language': self.language
        }