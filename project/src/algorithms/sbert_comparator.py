"""
Implementação do algoritmo SBERT (Sentence-BERT) para análise semântica.
Utiliza modelos pré-treinados para gerar embeddings de sentenças.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch

from src.utils.text_preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SBERTComparator:
    """
    Implementa comparação de textos usando SBERT (Sentence-BERT).
    
    SBERT é uma modificação do BERT que produz embeddings de sentenças
    semanticamente significativos que podem ser comparados usando
    similaridade de cosseno.
    """
    
    RECOMMENDED_MODELS = {
        'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
        'portuguese': 'neuralmind/bert-base-portuguese-cased',
        'english': 'all-MiniLM-L6-v2',
        'fast': 'all-MiniLM-L6-v2',
        'accurate': 'all-mpnet-base-v2'
    }
    
    def __init__(self, 
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 language: str = 'portuguese',
                 cache_folder: Optional[str] = None,
                 device: Optional[str] = None,
                 use_preprocessing: bool = True):
        """
        Inicializa o comparador SBERT.
        
        Args:
            model_name: Nome do modelo SBERT a usar
            language: Idioma principal dos textos
            cache_folder: Pasta para cache dos modelos
            device: Dispositivo para computação ('cpu', 'cuda')
            use_preprocessing: Usar pré-processamento de texto
        """
        self.model_name = model_name
        self.language = language
        self.cache_folder = cache_folder or str(Path.home() / '.cache' / 'sentence_transformers')
        self.use_preprocessing = use_preprocessing
        
        # Configura dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializa componentes
        self.model = None
        self.preprocessor = TextPreprocessor(language=language) if use_preprocessing else None
        self.embeddings_cache = {}
        self.comparison_results = []
        
        # Carrega modelo
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo SBERT."""
        try:
            logger.info(f"Carregando modelo SBERT: {self.model_name}")
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device
            )
            
            # Informações do modelo
            max_seq_length = getattr(self.model, 'max_seq_length', 512)
            logger.info(f"Modelo carregado. Comprimento máximo de sequência: {max_seq_length}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {self.model_name}: {str(e)}")
            # Fallback para modelo mais simples
            fallback_model = 'all-MiniLM-L6-v2'
            logger.info(f"Tentando modelo fallback: {fallback_model}")
            
            try:
                self.model = SentenceTransformer(
                    fallback_model,
                    cache_folder=self.cache_folder,
                    device=self.device
                )
                self.model_name = fallback_model
                logger.info("Modelo fallback carregado com sucesso")
            except Exception as e2:
                logger.error(f"Erro ao carregar modelo fallback: {str(e2)}")
                raise RuntimeError(f"Não foi possível carregar nenhum modelo SBERT: {str(e2)}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Aplica pré-processamento ao texto para embeddings.
        
        Args:
            text: Texto a ser processado
            
        Returns:
            str: Texto pré-processado
        """
        if not self.use_preprocessing or not self.preprocessor:
            return text
        
        return self.preprocessor.preprocess_for_embeddings(text)
    
    def encode_texts(self, texts: Union[str, List[str]], 
                    batch_size: int = 32,
                    show_progress: bool = True,
                    normalize_embeddings: bool = True) -> np.ndarray:
        """
        Codifica textos em embeddings.
        
        Args:
            texts: Texto ou lista de textos
            batch_size: Tamanho do batch para processamento
            show_progress: Mostrar barra de progresso
            normalize_embeddings: Normalizar embeddings
            
        Returns:
            np.ndarray: Array de embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Pré-processamento
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Verifica cache
        cache_keys = [hash(text) for text in processed_texts]
        cached_embeddings = []
        texts_to_encode = []
        cache_indices = []
        
        for i, (text, cache_key) in enumerate(zip(processed_texts, cache_keys)):
            if cache_key in self.embeddings_cache:
                cached_embeddings.append((i, self.embeddings_cache[cache_key]))
            else:
                texts_to_encode.append(text)
                cache_indices.append((i, cache_key))
        
        # Codifica textos não cacheados
        new_embeddings = []
        if texts_to_encode:
            logger.info(f"Codificando {len(texts_to_encode)} textos")
            
            try:
                embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True
                )
                
                # Atualiza cache
                for (original_idx, cache_key), embedding in zip(cache_indices, embeddings):
                    self.embeddings_cache[cache_key] = embedding
                    new_embeddings.append((original_idx, embedding))
                    
            except Exception as e:
                logger.error(f"Erro na codificação: {str(e)}")
                raise
        
        # Combina embeddings cacheados e novos
        all_embeddings = cached_embeddings + new_embeddings
        all_embeddings.sort(key=lambda x: x[0])  # Ordena por índice original
        
        result = np.array([embedding for _, embedding in all_embeddings])
        
        logger.info(f"Embeddings gerados: {result.shape}")
        return result
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """
        Compara dois textos usando SBERT e similaridade de cosseno.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Dict: Resultados da comparação
        """
        if not text1 or not text2:
            raise ValueError("Textos não podem estar vazios")
        
        # Gera embeddings
        embeddings = self.encode_texts([text1, text2])
        
        # Calcula similaridade de cosseno
        similarity_matrix = cosine_similarity(embeddings)
        similarity_score = similarity_matrix[0, 1]
        
        # Análise detalhada
        analysis = self._detailed_analysis(text1, text2, embeddings)
        
        result = {
            'similarity_score': float(similarity_score),
            'similarity_percentage': float(similarity_score * 100),
            'algorithm': f'SBERT ({self.model_name})',
            'text1_length': len(text1),
            'text2_length': len(text2),
            'embedding_dimension': embeddings.shape[1],
            'analysis': analysis,
            'interpretation': self._interpret_similarity(similarity_score)
        }
        
        # Armazena resultado
        self.comparison_results.append(result)
        
        return result
    
    def _detailed_analysis(self, text1: str, text2: str, embeddings: np.ndarray) -> Dict:
        """
        Realiza análise detalhada da comparação.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            embeddings: Embeddings dos textos
            
        Returns:
            Dict: Análise detalhada
        """
        # Estatísticas dos textos
        if self.preprocessor:
            stats1 = self.preprocessor.get_text_statistics(text1)
            stats2 = self.preprocessor.get_text_statistics(text2)
        else:
            stats1 = {'total_words': len(text1.split())}
            stats2 = {'total_words': len(text2.split())}
        
        # Análise dos embeddings
        embedding1, embedding2 = embeddings[0], embeddings[1]
        
        # Distância euclidiana
        euclidean_distance = float(np.linalg.norm(embedding1 - embedding2))
        
        # Correlação de Pearson
        correlation = float(np.corrcoef(embedding1, embedding2)[0, 1])
        
        # Análise de componentes principais (top dimensões)
        top_dimensions = self._analyze_top_dimensions(embedding1, embedding2)
        
        # Similaridade semântica por sentenças (se texto longo)
        sentence_similarities = self._analyze_sentence_similarities(text1, text2)
        
        return {
            'text1_stats': stats1,
            'text2_stats': stats2,
            'euclidean_distance': euclidean_distance,
            'pearson_correlation': correlation,
            'top_dimensions': top_dimensions,
            'sentence_similarities': sentence_similarities,
            'embedding_stats': {
                'mean_embedding1': float(np.mean(embedding1)),
                'std_embedding1': float(np.std(embedding1)),
                'mean_embedding2': float(np.mean(embedding2)),
                'std_embedding2': float(np.std(embedding2))
            }
        }
    
    def _analyze_top_dimensions(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                               top_n: int = 10) -> List[Dict]:
        """
        Analisa as dimensões mais importantes dos embeddings.
        
        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding
            top_n: Número de dimensões a analisar
            
        Returns:
            List[Dict]: Análise das dimensões
        """
        # Calcula diferenças absolutas
        differences = np.abs(embedding1 - embedding2)
        
        # Encontra as dimensões com maiores diferenças
        top_indices = np.argsort(differences)[-top_n:][::-1]
        
        result = []
        for idx in top_indices:
            result.append({
                'dimension': int(idx),
                'value1': float(embedding1[idx]),
                'value2': float(embedding2[idx]),
                'difference': float(differences[idx])
            })
        
        return result
    
    def _analyze_sentence_similarities(self, text1: str, text2: str) -> Dict:
        """
        Analisa similaridade entre sentenças dos textos.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Dict: Análise de similaridade por sentenças
        """
        if not self.preprocessor:
            return {}
        
        # Tokeniza em sentenças
        sentences1 = self.preprocessor.tokenize(text1, method='sentence')
        sentences2 = self.preprocessor.tokenize(text2, method='sentence')
        
        if len(sentences1) <= 1 or len(sentences2) <= 1:
            return {'note': 'Textos muito curtos para análise por sentenças'}
        
        # Limita número de sentenças para performance
        max_sentences = 5
        sentences1 = sentences1[:max_sentences]
        sentences2 = sentences2[:max_sentences]
        
        try:
            # Gera embeddings das sentenças
            all_sentences = sentences1 + sentences2
            sentence_embeddings = self.encode_texts(all_sentences, show_progress=False)
            
            # Separa embeddings
            emb1 = sentence_embeddings[:len(sentences1)]
            emb2 = sentence_embeddings[len(sentences1):]
            
            # Calcula matriz de similaridade
            similarity_matrix = cosine_similarity(emb1, emb2)
            
            # Encontra melhor match para cada sentença
            best_matches = []
            for i, sentence1 in enumerate(sentences1):
                best_idx = np.argmax(similarity_matrix[i])
                best_score = similarity_matrix[i, best_idx]
                
                best_matches.append({
                    'sentence1': sentence1[:100] + '...' if len(sentence1) > 100 else sentence1,
                    'sentence2': sentences2[best_idx][:100] + '...' if len(sentences2[best_idx]) > 100 else sentences2[best_idx],
                    'similarity': float(best_score)
                })
            
            return {
                'sentence_count1': len(sentences1),
                'sentence_count2': len(sentences2),
                'avg_similarity': float(np.mean(similarity_matrix)),
                'max_similarity': float(np.max(similarity_matrix)),
                'min_similarity': float(np.min(similarity_matrix)),
                'best_matches': best_matches[:3]  # Top 3 matches
            }
            
        except Exception as e:
            logger.warning(f"Erro na análise por sentenças: {str(e)}")
            return {'error': 'Erro na análise por sentenças'}
    
    def _interpret_similarity(self, score: float) -> str:
        """
        Interpreta o score de similaridade semântica.
        
        Args:
            score: Score de similaridade (0-1)
            
        Returns:
            str: Interpretação textual
        """
        if score >= 0.85:
            return "Textos semanticamente muito similares"
        elif score >= 0.70:
            return "Textos semanticamente similares"
        elif score >= 0.50:
            return "Textos com similaridade semântica moderada"
        elif score >= 0.30:
            return "Textos com baixa similaridade semântica"
        else:
            return "Textos semanticamente muito diferentes"
    
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
        
        # Gera embeddings
        embeddings = self.encode_texts(texts)
        
        # Calcula matriz de similaridade
        similarity_matrix = cosine_similarity(embeddings)
        
        # Cria DataFrame
        df = pd.DataFrame(
            similarity_matrix,
            index=[f"Texto_{i+1}" for i in range(len(texts))],
            columns=[f"Texto_{i+1}" for i in range(len(texts))]
        )
        
        return df
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Encontra textos mais similares a um texto de consulta.
        
        Args:
            query_text: Texto de consulta
            candidate_texts: Lista de textos candidatos
            top_k: Número de resultados a retornar
            
        Returns:
            List[Tuple[int, str, float]]: Lista de (índice, texto, similaridade)
        """
        if not candidate_texts:
            return []
        
        # Gera embeddings
        all_texts = [query_text] + candidate_texts
        embeddings = self.encode_texts(all_texts)
        
        # Calcula similaridades
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Ordena por similaridade
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in sorted_indices[:top_k]:
            results.append((
                i,
                candidate_texts[i][:200] + '...' if len(candidate_texts[i]) > 200 else candidate_texts[i],
                float(similarities[i])
            ))
        
        return results
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict:
        """
        Agrupa textos por similaridade semântica.
        
        Args:
            texts: Lista de textos
            n_clusters: Número de clusters
            
        Returns:
            Dict: Resultados do clustering
        """
        if len(texts) < n_clusters:
            raise ValueError(f"Número de textos ({len(texts)}) menor que clusters ({n_clusters})")
        
        from sklearn.cluster import KMeans
        
        # Gera embeddings
        embeddings = self.encode_texts(texts)
        
        # Aplica K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organiza resultados
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'index': i,
                'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                'full_text': texts[i]
            })
        
        # Calcula centróides e textos mais representativos
        cluster_info = {}
        for label, cluster_texts in clusters.items():
            cluster_embeddings = embeddings[[item['index'] for item in cluster_texts]]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Encontra texto mais próximo do centróide
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            representative_idx = np.argmin(distances)
            
            cluster_info[label] = {
                'size': len(cluster_texts),
                'texts': cluster_texts,
                'representative_text': cluster_texts[representative_idx]['text'],
                'avg_distance_to_centroid': float(np.mean(distances))
            }
        
        return {
            'clusters': cluster_info,
            'n_clusters': n_clusters,
            'total_texts': len(texts)
        }
    
    def get_model_info(self) -> Dict:
        """
        Obtém informações sobre o modelo.
        
        Returns:
            Dict: Informações do modelo
        """
        if not self.model:
            return {'status': 'not_loaded'}
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'language': self.language,
            'cache_size': len(self.embeddings_cache),
            'comparisons_made': len(self.comparison_results)
        }
    
    def clear_cache(self):
        """Limpa o cache de embeddings."""
        self.embeddings_cache.clear()
        logger.info("Cache de embeddings limpo")
    
    def save_embeddings(self, texts: List[str], filepath: str):
        """
        Salva embeddings de textos em arquivo.
        
        Args:
            texts: Lista de textos
            filepath: Caminho para salvar
        """
        embeddings = self.encode_texts(texts)
        
        data = {
            'texts': texts,
            'embeddings': embeddings,
            'model_name': self.model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        np.savez_compressed(filepath, **data)
        logger.info(f"Embeddings salvos em {filepath}")
    
    def load_embeddings(self, filepath: str) -> Tuple[List[str], np.ndarray]:
        """
        Carrega embeddings de arquivo.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Tuple[List[str], np.ndarray]: Textos e embeddings
        """
        data = np.load(filepath, allow_pickle=True)
        
        texts = data['texts'].tolist()
        embeddings = data['embeddings']
        
        logger.info(f"Embeddings carregados de {filepath}")
        return texts, embeddings