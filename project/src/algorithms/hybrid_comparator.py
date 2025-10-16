import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd

from src.algorithms.tfidf_comparator import TFIDFComparator
from src.algorithms.sbert_comparator import SBERTComparator
from src.utils.text_preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridComparator:
    """
    Implementa comparação híbrida adaptativa combinando TF-IDF e SBERT.
    
    O algoritmo ajusta automaticamente os pesos baseado nas características
    dos textos de entrada:
    - Textos técnicos/longos: Maior peso para TF-IDF
    - Textos curtos/conversacionais: Maior peso para SBERT
    - Ponderação dinâmica baseada em concordância entre métodos
    """
    
    def __init__(self,
                 language: str = 'portuguese',
                 sbert_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 default_tfidf_weight: float = 0.4,
                 default_sbert_weight: float = 0.6,
                 adaptive_weighting: bool = True,
                 min_text_length_for_tfidf: int = 100):
        """
        Inicializa o comparador híbrido.
        
        Args:
            language: Idioma dos textos
            sbert_model: Modelo SBERT a usar
            default_tfidf_weight: Peso padrão para TF-IDF
            default_sbert_weight: Peso padrão para SBERT
            adaptive_weighting: Usar ponderação adaptativa
            min_text_length_for_tfidf: Comprimento mínimo para usar TF-IDF
        """
        self.language = language
        self.sbert_model = sbert_model
        self.default_tfidf_weight = default_tfidf_weight
        self.default_sbert_weight = default_sbert_weight
        self.adaptive_weighting = adaptive_weighting
        self.min_text_length_for_tfidf = min_text_length_for_tfidf
        
        # Inicializa comparadores
        self.tfidf_comparator = TFIDFComparator(language=language)
        self.sbert_comparator = SBERTComparator(
            model_name=sbert_model,
            language=language
        )
        
        # Inicializa pré-processador
        self.preprocessor = TextPreprocessor(language=language)
        
        # Histórico de comparações
        self.comparison_results = []
        
        # Estatísticas de performance
        self.performance_stats = {
            'total_comparisons': 0,
            'avg_tfidf_weight': 0.0,
            'avg_sbert_weight': 0.0,
            'concordance_rate': 0.0
        }
        
        logger.info(f"Comparador híbrido inicializado com pesos padrão: "
                   f"TF-IDF={default_tfidf_weight}, SBERT={default_sbert_weight}")
    
    def _analyze_text_characteristics(self, text: str) -> Dict:
        """
        Analisa características do texto para ponderação adaptativa.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Dict: Características do texto
        """
        stats = self.preprocessor.get_text_statistics(text)
        
        # Características básicas
        length = stats.get('total_characters', 0)
        word_count = stats.get('total_words', 0)
        sentence_count = stats.get('total_sentences', 1)
        
        # Métricas derivadas
        avg_word_length = stats.get('avg_word_length', 0)
        avg_sentence_length = stats.get('avg_sentence_length', 0)
        vocabulary_richness = stats.get('vocabulary_richness', 0)
        
        # Classificação do tipo de texto
        text_type = self._classify_text_type(stats)
        
        return {
            'length': length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'vocabulary_richness': vocabulary_richness,
            'text_type': text_type,
            'is_technical': self._is_technical_text(text, stats),
            'complexity_score': self._calculate_complexity_score(stats)
        }
    
    def _classify_text_type(self, stats: Dict) -> str:
        """
        Classifica o tipo de texto baseado em estatísticas.
        
        Args:
            stats: Estatísticas do texto
            
        Returns:
            str: Tipo do texto
        """
        word_count = stats.get('total_words', 0)
        avg_sentence_length = stats.get('avg_sentence_length', 0)
        vocabulary_richness = stats.get('vocabulary_richness', 0)
        
        if word_count < 50:
            return 'very_short'
        elif word_count < 200:
            return 'short'
        elif word_count < 1000:
            return 'medium'
        else:
            return 'long'
    
    def _is_technical_text(self, text: str, stats: Dict) -> bool:
        """
        Determina se o texto é técnico baseado em características.
        
        Args:
            text: Texto original
            stats: Estatísticas do texto
            
        Returns:
            bool: True se o texto é técnico
        """
        # Indicadores de texto técnico
        technical_indicators = 0
        
        # Vocabulário rico (muitas palavras únicas)
        if stats.get('vocabulary_richness', 0) > 0.6:
            technical_indicators += 1
        
        # Sentenças longas
        if stats.get('avg_sentence_length', 0) > 20:
            technical_indicators += 1
        
        # Palavras longas
        if stats.get('avg_word_length', 0) > 6:
            technical_indicators += 1
        
        # Presença de termos técnicos comuns
        technical_terms = [
            'algoritmo', 'sistema', 'método', 'processo', 'análise',
            'implementação', 'desenvolvimento', 'arquitetura', 'framework',
            'biblioteca', 'função', 'classe', 'objeto', 'variável'
        ]
        
        text_lower = text.lower()
        technical_term_count = sum(1 for term in technical_terms if term in text_lower)
        
        if technical_term_count >= 3:
            technical_indicators += 1
        
        return technical_indicators >= 2
    
    def _calculate_complexity_score(self, stats: Dict) -> float:
        """
        Calcula score de complexidade do texto.
        
        Args:
            stats: Estatísticas do texto
            
        Returns:
            float: Score de complexidade (0-1)
        """
        # Normaliza métricas
        word_count_norm = min(stats.get('total_words', 0) / 1000, 1.0)
        vocab_richness = stats.get('vocabulary_richness', 0)
        avg_word_length_norm = min(stats.get('avg_word_length', 0) / 10, 1.0)
        avg_sentence_length_norm = min(stats.get('avg_sentence_length', 0) / 30, 1.0)
        
        # Combina métricas
        complexity = (
            word_count_norm * 0.2 +
            vocab_richness * 0.3 +
            avg_word_length_norm * 0.25 +
            avg_sentence_length_norm * 0.25
        )
        
        return min(complexity, 1.0)
    
    def _calculate_adaptive_weights(self, text1: str, text2: str) -> Tuple[float, float]:
        """
        Calcula pesos adaptativos baseado nas características dos textos.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Tuple[float, float]: Pesos (TF-IDF, SBERT)
        """
        if not self.adaptive_weighting:
            return self.default_tfidf_weight, self.default_sbert_weight
        
        # Analisa características dos textos
        chars1 = self._analyze_text_characteristics(text1)
        chars2 = self._analyze_text_characteristics(text2)
        
        # Combina características
        avg_length = (chars1['length'] + chars2['length']) / 2
        avg_complexity = (chars1['complexity_score'] + chars2['complexity_score']) / 2
        both_technical = chars1['is_technical'] and chars2['is_technical']
        
        # Calcula pesos baseado em regras
        tfidf_weight = self.default_tfidf_weight
        sbert_weight = self.default_sbert_weight
        
        # Ajustes baseados no comprimento
        if avg_length < self.min_text_length_for_tfidf:
            # Textos muito curtos: favorece SBERT
            tfidf_weight *= 0.5
            sbert_weight *= 1.5
        elif avg_length > 500:
            # Textos longos: favorece TF-IDF
            tfidf_weight *= 1.3
            sbert_weight *= 0.8
        
        # Ajustes baseados na complexidade
        if avg_complexity > 0.7:
            # Textos complexos: favorece TF-IDF
            tfidf_weight *= 1.2
            sbert_weight *= 0.9
        elif avg_complexity < 0.3:
            # Textos simples: favorece SBERT
            tfidf_weight *= 0.8
            sbert_weight *= 1.2
        
        # Ajustes para textos técnicos
        if both_technical:
            tfidf_weight *= 1.4
            sbert_weight *= 0.7
        
        # Normaliza pesos
        total_weight = tfidf_weight + sbert_weight
        tfidf_weight /= total_weight
        sbert_weight /= total_weight
        
        logger.debug(f"Pesos adaptativos: TF-IDF={tfidf_weight:.3f}, SBERT={sbert_weight:.3f}")
        
        return tfidf_weight, sbert_weight
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """
        Compara dois textos usando abordagem híbrida.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Dict: Resultados da comparação híbrida
        """
        if not text1 or not text2:
            raise ValueError("Textos não podem estar vazios")
        
        logger.info("Iniciando comparação híbrida")
        
        # Calcula pesos adaptativos
        tfidf_weight, sbert_weight = self._calculate_adaptive_weights(text1, text2)
        
        # Executa comparações individuais
        tfidf_result = self.tfidf_comparator.compare_texts(text1, text2)
        sbert_result = self.sbert_comparator.compare_texts(text1, text2)
        
        # Calcula similaridade híbrida
        tfidf_score = tfidf_result['similarity_score']
        sbert_score = sbert_result['similarity_score']
        
        hybrid_score = (tfidf_weight * tfidf_score) + (sbert_weight * sbert_score)
        
        # Análise de concordância
        concordance_analysis = self._analyze_concordance(tfidf_score, sbert_score)
        
        # Análise detalhada
        detailed_analysis = self._detailed_hybrid_analysis(
            text1, text2, tfidf_result, sbert_result, 
            tfidf_weight, sbert_weight
        )
        
        result = {
            'similarity_score': float(hybrid_score),
            'similarity_percentage': float(hybrid_score * 100),
            'algorithm': 'Hybrid (TF-IDF + SBERT)',
            'weights': {
                'tfidf_weight': float(tfidf_weight),
                'sbert_weight': float(sbert_weight)
            },
            'individual_scores': {
                'tfidf_score': float(tfidf_score),
                'sbert_score': float(sbert_score)
            },
            'concordance': concordance_analysis,
            'text1_length': len(text1),
            'text2_length': len(text2),
            'analysis': detailed_analysis,
            'interpretation': self._interpret_hybrid_similarity(
                hybrid_score, tfidf_score, sbert_score, concordance_analysis
            )
        }
        
        # Atualiza estatísticas
        self._update_performance_stats(tfidf_weight, sbert_weight, concordance_analysis)
        
        # Armazena resultado
        self.comparison_results.append(result)
        
        return result
    
    def _analyze_concordance(self, tfidf_score: float, sbert_score: float) -> Dict:
        """
        Analisa concordância entre os métodos.
        
        Args:
            tfidf_score: Score TF-IDF
            sbert_score: Score SBERT
            
        Returns:
            Dict: Análise de concordância
        """
        difference = abs(tfidf_score - sbert_score)
        
        # Classifica concordância
        if difference <= 0.1:
            concordance_level = 'high'
            concordance_description = 'Alta concordância entre métodos'
        elif difference <= 0.3:
            concordance_level = 'medium'
            concordance_description = 'Concordância moderada entre métodos'
        else:
            concordance_level = 'low'
            concordance_description = 'Baixa concordância entre métodos'
        
        # Determina qual método é mais confiável
        if difference > 0.2:
            if tfidf_score > sbert_score:
                preferred_method = 'tfidf'
                confidence_note = 'TF-IDF indica maior similaridade lexical'
            else:
                preferred_method = 'sbert'
                confidence_note = 'SBERT indica maior similaridade semântica'
        else:
            preferred_method = 'both'
            confidence_note = 'Ambos os métodos concordam'
        
        return {
            'difference': float(difference),
            'level': concordance_level,
            'description': concordance_description,
            'preferred_method': preferred_method,
            'confidence_note': confidence_note,
            'correlation': float(np.corrcoef([tfidf_score], [sbert_score])[0, 1])
        }
    
    def _detailed_hybrid_analysis(self, text1: str, text2: str,
                                 tfidf_result: Dict, sbert_result: Dict,
                                 tfidf_weight: float, sbert_weight: float) -> Dict:
        """
        Realiza análise detalhada da comparação híbrida.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            tfidf_result: Resultado TF-IDF
            sbert_result: Resultado SBERT
            tfidf_weight: Peso TF-IDF
            sbert_weight: Peso SBERT
            
        Returns:
            Dict: Análise detalhada
        """
        # Características dos textos
        chars1 = self._analyze_text_characteristics(text1)
        chars2 = self._analyze_text_characteristics(text2)
        
        # Combina análises individuais
        combined_analysis = {
            'text_characteristics': {
                'text1': chars1,
                'text2': chars2
            },
            'tfidf_analysis': tfidf_result.get('analysis', {}),
            'sbert_analysis': sbert_result.get('analysis', {}),
            'weight_justification': self._explain_weight_selection(
                chars1, chars2, tfidf_weight, sbert_weight
            )
        }
        
        return combined_analysis
    
    def _explain_weight_selection(self, chars1: Dict, chars2: Dict,
                                 tfidf_weight: float, sbert_weight: float) -> str:
        """
        Explica a seleção de pesos.
        
        Args:
            chars1: Características do texto 1
            chars2: Características do texto 2
            tfidf_weight: Peso TF-IDF
            sbert_weight: Peso SBERT
            
        Returns:
            str: Explicação da seleção de pesos
        """
        explanations = []
        
        avg_length = (chars1['length'] + chars2['length']) / 2
        both_technical = chars1['is_technical'] and chars2['is_technical']
        avg_complexity = (chars1['complexity_score'] + chars2['complexity_score']) / 2
        
        if tfidf_weight > sbert_weight:
            explanations.append("TF-IDF recebeu maior peso devido a:")
            
            if avg_length > 500:
                explanations.append("- Textos longos (melhor para análise lexical)")
            
            if both_technical:
                explanations.append("- Textos técnicos (vocabulário específico)")
            
            if avg_complexity > 0.7:
                explanations.append("- Alta complexidade textual")
        
        else:
            explanations.append("SBERT recebeu maior peso devido a:")
            
            if avg_length < self.min_text_length_for_tfidf:
                explanations.append("- Textos curtos (melhor para análise semântica)")
            
            if not both_technical:
                explanations.append("- Textos não-técnicos (contexto semântico)")
            
            if avg_complexity < 0.3:
                explanations.append("- Baixa complexidade textual")
        
        return " ".join(explanations) if explanations else "Pesos padrão aplicados"
    
    def _interpret_hybrid_similarity(self, hybrid_score: float, 
                                   tfidf_score: float, sbert_score: float,
                                   concordance: Dict) -> str:
        """
        Interpreta o resultado da similaridade híbrida.
        
        Args:
            hybrid_score: Score híbrido
            tfidf_score: Score TF-IDF
            sbert_score: Score SBERT
            concordance: Análise de concordância
            
        Returns:
            str: Interpretação textual
        """
        base_interpretation = self._get_base_interpretation(hybrid_score)
        
        # Adiciona contexto baseado na concordância
        if concordance['level'] == 'high':
            context = " (ambos os métodos concordam)"
        elif concordance['level'] == 'low':
            if concordance['preferred_method'] == 'tfidf':
                context = " (TF-IDF sugere maior similaridade lexical)"
            elif concordance['preferred_method'] == 'sbert':
                context = " (SBERT sugere maior similaridade semântica)"
            else:
                context = " (métodos apresentam divergência)"
        else:
            context = " (concordância moderada entre métodos)"
        
        return base_interpretation + context
    
    def _get_base_interpretation(self, score: float) -> str:
        """
        Interpretação base do score de similaridade.
        
        Args:
            score: Score de similaridade
            
        Returns:
            str: Interpretação base
        """
        if score >= 0.85:
            return "Textos muito similares"
        elif score >= 0.70:
            return "Textos similares"
        elif score >= 0.50:
            return "Textos moderadamente similares"
        elif score >= 0.30:
            return "Textos com baixa similaridade"
        else:
            return "Textos muito diferentes"
    
    def _update_performance_stats(self, tfidf_weight: float, sbert_weight: float,
                                 concordance: Dict):
        """
        Atualiza estatísticas de performance.
        
        Args:
            tfidf_weight: Peso TF-IDF usado
            sbert_weight: Peso SBERT usado
            concordance: Análise de concordância
        """
        self.performance_stats['total_comparisons'] += 1
        n = self.performance_stats['total_comparisons']
        
        # Atualiza médias
        self.performance_stats['avg_tfidf_weight'] = (
            (self.performance_stats['avg_tfidf_weight'] * (n-1) + tfidf_weight) / n
        )
        
        self.performance_stats['avg_sbert_weight'] = (
            (self.performance_stats['avg_sbert_weight'] * (n-1) + sbert_weight) / n
        )
        
        # Atualiza taxa de concordância
        concordance_score = 1.0 if concordance['level'] == 'high' else (
            0.5 if concordance['level'] == 'medium' else 0.0
        )
        
        self.performance_stats['concordance_rate'] = (
            (self.performance_stats['concordance_rate'] * (n-1) + concordance_score) / n
        )
    
    def batch_compare(self, texts: List[str]) -> pd.DataFrame:
        """
        Compara múltiplos textos usando abordagem híbrida.
        
        Args:
            texts: Lista de textos para comparar
            
        Returns:
            pd.DataFrame: Matriz de similaridades híbridas
        """
        if len(texts) < 2:
            raise ValueError("Necessário pelo menos 2 textos para comparação")
        
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        # Compara todos os pares
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    result = self.compare_texts(texts[i], texts[j])
                    similarity = result['similarity_score']
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Matriz simétrica
        
        # Cria DataFrame
        df = pd.DataFrame(
            similarity_matrix,
            index=[f"Texto_{i+1}" for i in range(n)],
            columns=[f"Texto_{i+1}" for i in range(n)]
        )
        
        return df
    
    def get_performance_stats(self) -> Dict:
        """
        Obtém estatísticas de performance do comparador.
        
        Returns:
            Dict: Estatísticas de performance
        """
        return {
            **self.performance_stats,
            'tfidf_model_stats': self.tfidf_comparator.get_statistics(),
            'sbert_model_stats': self.sbert_comparator.get_model_info()
        }
    
    def optimize_weights(self, validation_data: List[Tuple[str, str, float]]) -> Dict:
        """
        Otimiza pesos usando dados de validação.
        
        Args:
            validation_data: Lista de (texto1, texto2, similaridade_esperada)
            
        Returns:
            Dict: Resultados da otimização
        """
        from sklearn.metrics import mean_squared_error
        from scipy.optimize import minimize
        
        def objective(weights):
            tfidf_w, sbert_w = weights
            total_w = tfidf_w + sbert_w
            tfidf_w, sbert_w = tfidf_w/total_w, sbert_w/total_w
            
            predictions = []
            for text1, text2, expected in validation_data:
                tfidf_score = self.tfidf_comparator.compare_texts(text1, text2)['similarity_score']
                sbert_score = self.sbert_comparator.compare_texts(text1, text2)['similarity_score']
                hybrid_score = tfidf_w * tfidf_score + sbert_w * sbert_score
                predictions.append(hybrid_score)
            
            expected_scores = [expected for _, _, expected in validation_data]
            return mean_squared_error(expected_scores, predictions)
        
        # Otimização
        result = minimize(
            objective,
            x0=[self.default_tfidf_weight, self.default_sbert_weight],
            bounds=[(0.1, 0.9), (0.1, 0.9)],
            method='L-BFGS-B'
        )
        
        if result.success:
            optimal_weights = result.x
            total_w = sum(optimal_weights)
            optimal_tfidf = optimal_weights[0] / total_w
            optimal_sbert = optimal_weights[1] / total_w
            
            # Atualiza pesos padrão
            self.default_tfidf_weight = optimal_tfidf
            self.default_sbert_weight = optimal_sbert
            
            logger.info(f"Pesos otimizados: TF-IDF={optimal_tfidf:.3f}, SBERT={optimal_sbert:.3f}")
            
            return {
                'success': True,
                'optimal_tfidf_weight': optimal_tfidf,
                'optimal_sbert_weight': optimal_sbert,
                'mse': result.fun,
                'validation_size': len(validation_data)
            }
        else:
            logger.warning("Otimização de pesos falhou")
            return {
                'success': False,
                'error': result.message
            }
    
    def explain_comparison(self, text1: str, text2: str) -> str:
        """
        Gera explicação detalhada de uma comparação.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            str: Explicação detalhada
        """
        result = self.compare_texts(text1, text2)
        
        explanation = f"""
        ANÁLISE HÍBRIDA DE SIMILARIDADE TEXTUAL
        
        Similaridade Final: {result['similarity_percentage']:.1f}%
        {result['interpretation']}
        
        PESOS UTILIZADOS:
        - TF-IDF: {result['weights']['tfidf_weight']:.1%}
        - SBERT: {result['weights']['sbert_weight']:.1%}
        
        SCORES INDIVIDUAIS:
        - TF-IDF (análise lexical): {result['individual_scores']['tfidf_score']:.3f}
        - SBERT (análise semântica): {result['individual_scores']['sbert_score']:.3f}
        
        CONCORDÂNCIA ENTRE MÉTODOS:
        {result['concordance']['description']}
        Diferença: {result['concordance']['difference']:.3f}
        
        JUSTIFICATIVA DOS PESOS:
        {result['analysis']['weight_justification']}
        """
        
        return explanation.strip()