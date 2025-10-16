"""
Testes unitários para o módulo TFIDFComparator.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from algorithms.tfidf_comparator import TFIDFComparator


class TestTFIDFComparator:
    """Testes para a classe TFIDFComparator."""
    
    def setup_method(self):
        """Setup executado antes de cada teste."""
        self.comparator = TFIDFComparator(language='portuguese')
    
    def test_init_default(self):
        """Testa inicialização com parâmetros padrão."""
        comparator = TFIDFComparator()
        
        assert comparator.language == 'portuguese'
        assert comparator.max_features is None
        assert comparator.min_df == 1
        assert comparator.max_df == 1.0
        assert comparator.ngram_range == (1, 2)
        assert comparator.use_custom_preprocessing is True
        assert comparator.vectorizer is None
        assert comparator.comparison_results == []
    
    def test_init_custom_params(self):
        """Testa inicialização com parâmetros customizados."""
        comparator = TFIDFComparator(
            language='english',
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 3),
            use_custom_preprocessing=False
        )
        
        assert comparator.language == 'english'
        assert comparator.max_features == 1000
        assert comparator.min_df == 2
        assert comparator.max_df == 0.8
        assert comparator.ngram_range == (1, 3)
        assert comparator.use_custom_preprocessing is False
    
    def test_preprocess_text_custom(self):
        """Testa pré-processamento customizado."""
        text = "Este é um TESTE com Acentos!"
        processed = self.comparator._preprocess_text(text)
        
        # Deve estar em minúsculas
        assert processed.islower()
        # Deve conter o texto processado
        assert "teste" in processed
        assert "acentos" in processed
    
    def test_preprocess_text_disabled(self):
        """Testa pré-processamento desabilitado."""
        comparator = TFIDFComparator(use_custom_preprocessing=False)
        text = "Este é um TESTE com Acentos!"
        processed = comparator._preprocess_text(text)
        
        # Deve apenas converter para minúsculas
        assert processed == text.lower()
    
    def test_fit_simple(self):
        """Testa treinamento com textos simples."""
        texts = [
            "Este é o primeiro texto de teste",
            "Este é o segundo texto para comparação",
            "Terceiro texto completamente diferente"
        ]
        
        result = self.comparator.fit(texts)
        
        # Deve retornar self para chaining
        assert result is self.comparator
        
        # Deve ter criado o vectorizer
        assert self.comparator.vectorizer is not None
        assert self.comparator.corpus_vectors is not None
        assert self.comparator.feature_names is not None
        assert len(self.comparator.corpus_texts) == 3
    
    def test_fit_empty_texts(self):
        """Testa treinamento com lista vazia."""
        with pytest.raises(ValueError, match="Lista de textos não pode estar vazia"):
            self.comparator.fit([])
    
    def test_transform_without_fit(self):
        """Testa transformação sem treinamento prévio."""
        with pytest.raises(ValueError, match="Modelo não foi treinado"):
            self.comparator.transform(["texto teste"])
    
    def test_transform_after_fit(self):
        """Testa transformação após treinamento."""
        # Treina modelo
        train_texts = ["primeiro texto", "segundo texto"]
        self.comparator.fit(train_texts)
        
        # Transforma novos textos
        test_texts = ["novo texto", "outro texto"]
        vectors = self.comparator.transform(test_texts)
        
        assert vectors is not None
        assert vectors.shape[0] == 2  # 2 textos
        assert vectors.shape[1] > 0   # Deve ter features
    
    def test_compare_texts_simple(self):
        """Testa comparação simples entre dois textos."""
        text1 = "Este é um texto sobre algoritmos de ordenação"
        text2 = "Este texto fala sobre algoritmos de busca"
        
        result = self.comparator.compare_texts(text1, text2)
        
        # Verifica estrutura do resultado
        assert 'similarity_score' in result
        assert 'similarity_percentage' in result
        assert 'algorithm' in result
        assert 'interpretation' in result
        assert 'analysis' in result
        
        # Verifica valores
        assert 0 <= result['similarity_score'] <= 1
        assert 0 <= result['similarity_percentage'] <= 100
        assert result['algorithm'] == 'TF-IDF + Cosine Similarity'
        
        # Deve ter alta similaridade (palavras em comum)
        assert result['similarity_score'] > 0.3
    
    def test_compare_texts_identical(self):
        """Testa comparação de textos idênticos."""
        text = "Este é um texto de teste para comparação"
        
        result = self.comparator.compare_texts(text, text)
        
        # Textos idênticos devem ter similaridade máxima
        assert result['similarity_score'] == pytest.approx(1.0, abs=0.01)
        assert result['similarity_percentage'] == pytest.approx(100.0, abs=1.0)
    
    def test_compare_texts_completely_different(self):
        """Testa comparação de textos completamente diferentes."""
        text1 = "algoritmo ordenação quicksort mergesort"
        text2 = "culinária italiana pizza pasta risotto"
        
        result = self.comparator.compare_texts(text1, text2)
        
        # Textos diferentes devem ter baixa similaridade
        assert result['similarity_score'] < 0.3
        assert result['similarity_percentage'] < 30.0
    
    def test_compare_texts_empty(self):
        """Testa comparação com textos vazios."""
        with pytest.raises(ValueError, match="Textos não podem estar vazios"):
            self.comparator.compare_texts("", "texto não vazio")
        
        with pytest.raises(ValueError, match="Textos não podem estar vazios"):
            self.comparator.compare_texts("texto não vazio", "")
        
        with pytest.raises(ValueError, match="Textos não podem estar vazios"):
            self.comparator.compare_texts("", "")
    
    def test_detailed_analysis_structure(self):
        """Testa estrutura da análise detalhada."""
        text1 = "análise algoritmo estrutura dados"
        text2 = "análise sistema processamento informação"
        
        result = self.comparator.compare_texts(text1, text2)
        analysis = result['analysis']
        
        # Verifica estrutura da análise
        assert 'text1_stats' in analysis
        assert 'text2_stats' in analysis
        assert 'top_terms_text1' in analysis
        assert 'top_terms_text2' in analysis
        assert 'common_terms' in analysis
        assert 'lexical_diversity' in analysis
        assert 'vocabulary_overlap' in analysis
    
    def test_get_top_terms(self):
        """Testa extração de termos mais importantes."""
        texts = ["algoritmo ordenação dados estrutura", "sistema processamento informação"]
        self.comparator.fit(texts)
        
        # Transforma primeiro texto
        vectors = self.comparator.transform([texts[0]])
        vector = vectors.toarray().flatten()
        
        top_terms = self.comparator._get_top_terms(vector, n=3)
        
        assert len(top_terms) <= 3
        for term, score in top_terms:
            assert isinstance(term, str)
            assert isinstance(score, float)
            assert score > 0
    
    def test_get_common_terms(self):
        """Testa extração de termos comuns."""
        texts = ["algoritmo análise dados", "algoritmo processamento dados"]
        self.comparator.fit(texts)
        
        vectors = self.comparator.transform(texts)
        vector1 = vectors[0].toarray().flatten()
        vector2 = vectors[1].toarray().flatten()
        
        common_terms = self.comparator._get_common_terms(vector1, vector2, n=5)
        
        # Deve encontrar termos comuns
        assert len(common_terms) > 0
        
        for term, score1, score2 in common_terms:
            assert isinstance(term, str)
            assert isinstance(score1, float)
            assert isinstance(score2, float)
            assert score1 > 0
            assert score2 > 0
    
    def test_calculate_lexical_diversity(self):
        """Testa cálculo de diversidade lexical."""
        text1 = "algoritmo ordenação dados estrutura"
        text2 = "algoritmo busca informação sistema"
        
        diversity = self.comparator._calculate_lexical_diversity(text1, text2)
        
        assert 'unique_tokens_text1' in diversity
        assert 'unique_tokens_text2' in diversity
        assert 'common_tokens' in diversity
        assert 'total_unique_tokens' in diversity
        assert 'jaccard_similarity' in diversity
        
        # Verifica valores
        assert diversity['unique_tokens_text1'] > 0
        assert diversity['unique_tokens_text2'] > 0
        assert 0 <= diversity['jaccard_similarity'] <= 1
    
    def test_calculate_vocabulary_overlap(self):
        """Testa cálculo de sobreposição de vocabulário."""
        text1 = "palavra comum teste"
        text2 = "palavra comum exemplo"
        
        overlap = self.comparator._calculate_vocabulary_overlap(text1, text2)
        
        assert 0 <= overlap <= 1
        # Deve ter sobreposição devido às palavras "palavra" e "comum"
        assert overlap > 0
    
    def test_interpret_similarity(self):
        """Testa interpretação de scores de similaridade."""
        # Testa diferentes faixas
        assert "muito similares" in self.comparator._interpret_similarity(0.95)
        assert "similares" in self.comparator._interpret_similarity(0.75)
        assert "moderadamente" in self.comparator._interpret_similarity(0.55)
        assert "baixa" in self.comparator._interpret_similarity(0.35)
        assert "muito diferentes" in self.comparator._interpret_similarity(0.15)
    
    def test_batch_compare(self):
        """Testa comparação em lote."""
        texts = [
            "primeiro texto sobre algoritmos",
            "segundo texto sobre estruturas",
            "terceiro texto sobre programação"
        ]
        
        similarity_matrix = self.comparator.batch_compare(texts)
        
        # Verifica estrutura
        assert isinstance(similarity_matrix, pd.DataFrame)
        assert similarity_matrix.shape == (3, 3)
        
        # Diagonal deve ser 1.0 (auto-similaridade)
        for i in range(3):
            assert similarity_matrix.iloc[i, i] == pytest.approx(1.0, abs=0.01)
        
        # Matriz deve ser simétrica
        for i in range(3):
            for j in range(3):
                assert similarity_matrix.iloc[i, j] == pytest.approx(
                    similarity_matrix.iloc[j, i], abs=0.01
                )
    
    def test_batch_compare_insufficient_texts(self):
        """Testa comparação em lote com textos insuficientes."""
        with pytest.raises(ValueError, match="Necessário pelo menos 2 textos"):
            self.comparator.batch_compare(["apenas um texto"])
    
    def test_get_feature_importance(self):
        """Testa obtenção de importância das features."""
        texts = ["algoritmo ordenação dados", "sistema processamento informação"]
        self.comparator.fit(texts)
        
        importance = self.comparator.get_feature_importance(texts[0], top_n=5)
        
        assert len(importance) <= 5
        for feature, score in importance:
            assert isinstance(feature, str)
            assert isinstance(score, float)
            assert score > 0
    
    def test_get_feature_importance_without_fit(self):
        """Testa obtenção de importância sem treinamento."""
        with pytest.raises(ValueError, match="Modelo não foi treinado"):
            self.comparator.get_feature_importance("texto teste")
    
    def test_get_statistics_not_trained(self):
        """Testa estatísticas de modelo não treinado."""
        stats = self.comparator.get_statistics()
        assert stats['status'] == 'not_trained'
    
    def test_get_statistics_trained(self):
        """Testa estatísticas de modelo treinado."""
        texts = ["texto um", "texto dois"]
        self.comparator.fit(texts)
        
        stats = self.comparator.get_statistics()
        
        assert stats['status'] == 'trained'
        assert 'vocabulary_size' in stats
        assert 'corpus_size' in stats
        assert 'comparisons_made' in stats
        assert 'ngram_range' in stats
        assert 'language' in stats
        
        assert stats['corpus_size'] == 2
        assert stats['language'] == 'portuguese'
    
    def test_comparison_results_storage(self):
        """Testa armazenamento de resultados de comparação."""
        text1 = "primeiro texto"
        text2 = "segundo texto"
        
        # Inicialmente vazio
        assert len(self.comparator.comparison_results) == 0
        
        # Executa comparação
        self.comparator.compare_texts(text1, text2)
        
        # Deve ter armazenado resultado
        assert len(self.comparator.comparison_results) == 1
        
        # Executa outra comparação
        self.comparator.compare_texts(text2, text1)
        
        # Deve ter dois resultados
        assert len(self.comparator.comparison_results) == 2


class TestTFIDFComparatorAdvanced:
    """Testes avançados para TFIDFComparator."""
    
    def setup_method(self):
        """Setup executado antes de cada teste."""
        self.comparator = TFIDFComparator(language='portuguese')
    
    def test_different_ngram_ranges(self):
        """Testa diferentes configurações de n-grams."""
        text1 = "algoritmo de ordenação rápida"
        text2 = "algoritmo de busca binária"
        
        # Testa unigrams apenas
        comparator_1gram = TFIDFComparator(ngram_range=(1, 1))
        result_1gram = comparator_1gram.compare_texts(text1, text2)
        
        # Testa bigrams
        comparator_2gram = TFIDFComparator(ngram_range=(1, 2))
        result_2gram = comparator_2gram.compare_texts(text1, text2)
        
        # Ambos devem funcionar
        assert 0 <= result_1gram['similarity_score'] <= 1
        assert 0 <= result_2gram['similarity_score'] <= 1
        
        # Bigrams podem capturar "algoritmo de" como termo comum
        # então pode ter similaridade diferente
    
    def test_max_features_limitation(self):
        """Testa limitação de features máximas."""
        # Cria textos com vocabulário grande
        texts = []
        for i in range(10):
            text = " ".join([f"palavra{j}" for j in range(i*10, (i+1)*10)])
            texts.append(text)
        
        # Comparador com limite de features
        comparator = TFIDFComparator(max_features=50)
        comparator.fit(texts)
        
        # Deve respeitar o limite
        assert len(comparator.feature_names) <= 50
    
    def test_min_df_filtering(self):
        """Testa filtragem por frequência mínima de documento."""
        texts = [
            "palavra comum texto um",
            "palavra comum texto dois", 
            "palavra comum texto três",
            "palavra rara apenas aqui"
        ]
        
        # min_df=2 deve filtrar "rara" e "apenas" e "aqui"
        comparator = TFIDFComparator(min_df=2)
        comparator.fit(texts)
        
        # Verifica se palavras raras foram filtradas
        feature_names = list(comparator.feature_names)
        assert "comum" in " ".join(feature_names)  # Deve estar presente
        # Palavras que aparecem apenas uma vez devem ser filtradas
    
    def test_max_df_filtering(self):
        """Testa filtragem por frequência máxima de documento."""
        texts = [
            "palavra muito comum em todos",
            "palavra muito comum em todos",
            "palavra muito comum em todos",
            "palavra específica apenas aqui"
        ]
        
        # max_df=0.5 deve filtrar palavras que aparecem em mais de 50% dos docs
        comparator = TFIDFComparator(max_df=0.5)
        comparator.fit(texts)
        
        # Palavras muito comuns devem ser filtradas
        feature_names = list(comparator.feature_names)
        # "específica" deve estar presente, palavras muito comuns podem ser filtradas
    
    def test_language_specific_stopwords(self):
        """Testa stopwords específicas do idioma."""
        # Português
        comparator_pt = TFIDFComparator(language='portuguese')
        text_pt = "Este é um texto em português com artigos"
        
        # Inglês
        comparator_en = TFIDFComparator(language='english')
        text_en = "This is a text in english with articles"
        
        # Ambos devem processar sem erro
        result_pt = comparator_pt.compare_texts(text_pt, text_pt)
        result_en = comparator_en.compare_texts(text_en, text_en)
        
        assert result_pt['similarity_score'] == pytest.approx(1.0, abs=0.01)
        assert result_en['similarity_score'] == pytest.approx(1.0, abs=0.01)
    
    def test_edge_case_single_word(self):
        """Testa caso extremo com palavras únicas."""
        result = self.comparator.compare_texts("algoritmo", "algoritmo")
        assert result['similarity_score'] == pytest.approx(1.0, abs=0.01)
        
        result = self.comparator.compare_texts("algoritmo", "estrutura")
        assert result['similarity_score'] == pytest.approx(0.0, abs=0.01)
    
    def test_edge_case_repeated_words(self):
        """Testa caso com palavras repetidas."""
        text1 = "algoritmo algoritmo algoritmo"
        text2 = "algoritmo estrutura dados"
        
        result = self.comparator.compare_texts(text1, text2)
        
        # Deve ter alguma similaridade devido à palavra comum
        assert result['similarity_score'] > 0
    
    def test_performance_with_long_texts(self):
        """Testa performance com textos longos."""
        # Cria textos longos
        long_text1 = " ".join(["palavra" + str(i) for i in range(1000)])
        long_text2 = " ".join(["palavra" + str(i) for i in range(500, 1500)])
        
        # Deve processar sem erro
        result = self.comparator.compare_texts(long_text1, long_text2)
        
        assert 'similarity_score' in result
        assert 0 <= result['similarity_score'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])