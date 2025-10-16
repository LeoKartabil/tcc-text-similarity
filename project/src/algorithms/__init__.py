"""
Módulo de algoritmos de comparação textual.
Implementa TF-IDF, SBERT e algoritmo híbrido.
"""

from src.algorithms.tfidf_comparator import TFIDFComparator
from src.algorithms.sbert_comparator import SBERTComparator
from src.algorithms.hybrid_comparator import HybridComparator

__all__ = ['TFIDFComparator', 'SBERTComparator', 'HybridComparator']