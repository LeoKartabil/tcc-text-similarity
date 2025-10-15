"""
Módulo para pré-processamento de texto.
Inclui limpeza, normalização e preparação de texto para análise.
"""

import re
import string
import logging
from typing import List, Optional, Set
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Classe responsável pelo pré-processamento de texto.
    
    Funcionalidades:
    - Limpeza de texto (remoção de caracteres especiais, normalização)
    - Tokenização
    - Remoção de stopwords
    - Stemming e lemmatização
    - Normalização de encoding
    """
    
    def __init__(self, language: str = 'portuguese'):
        """
        Inicializa o pré-processador de texto.
        
        Args:
            language: Idioma para stopwords ('portuguese', 'english')
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download necessário dos recursos NLTK
        self._download_nltk_resources()
        
        # Carrega stopwords
        self.stopwords = self._load_stopwords()
        
        # Padrões regex para limpeza
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'\b\d{2,3}[-.\s]?\d{4,5}[-.\s]?\d{4}\b'),
            'numbers': re.compile(r'\b\d+\b'),
            'special_chars': re.compile(r'[^\w\s]'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n+')
        }
    
    def _download_nltk_resources(self):
        """Download dos recursos necessários do NLTK."""
        resources = [
            'punkt', 'stopwords', 'wordnet', 'omw-1.4',
            'punkt_tab'
        ]
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    logger.info(f"Downloaded NLTK resource: {resource}")
                except Exception as e:
                    logger.warning(f"Failed to download {resource}: {str(e)}")
    
    def _load_stopwords(self) -> Set[str]:
        """
        Carrega stopwords para o idioma especificado.
        
        Returns:
            Set[str]: Conjunto de stopwords
        """
        try:
            if self.language == 'portuguese':
                stop_words = set(stopwords.words('portuguese'))
                # Adiciona stopwords customizadas em português
                custom_stopwords = {
                    'ser', 'estar', 'ter', 'haver', 'fazer', 'dizer', 'dar', 'ir',
                    'vir', 'ver', 'saber', 'poder', 'querer', 'ficar', 'pôr',
                    'muito', 'bem', 'já', 'mais', 'depois', 'dois', 'três',
                    'primeiro', 'segundo', 'terceiro', 'último', 'próximo'
                }
                stop_words.update(custom_stopwords)
            else:
                stop_words = set(stopwords.words('english'))
                
            logger.info(f"Loaded {len(stop_words)} stopwords for {self.language}")
            return stop_words
            
        except Exception as e:
            logger.warning(f"Failed to load stopwords: {str(e)}")
            return set()
    
    def clean_text(self, text: str, 
                   remove_emails: bool = True,
                   remove_urls: bool = True,
                   remove_phones: bool = True,
                   remove_numbers: bool = False,
                   remove_special_chars: bool = True,
                   normalize_whitespace: bool = True) -> str:
        """
        Limpa o texto removendo elementos indesejados.
        
        Args:
            text: Texto a ser limpo
            remove_emails: Remove endereços de email
            remove_urls: Remove URLs
            remove_phones: Remove números de telefone
            remove_numbers: Remove números
            remove_special_chars: Remove caracteres especiais
            normalize_whitespace: Normaliza espaços em branco
            
        Returns:
            str: Texto limpo
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normaliza encoding Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove elementos específicos
        if remove_emails:
            text = self.patterns['email'].sub('', text)
        
        if remove_urls:
            text = self.patterns['url'].sub('', text)
        
        if remove_phones:
            text = self.patterns['phone'].sub('', text)
        
        if remove_numbers:
            text = self.patterns['numbers'].sub('', text)
        
        if remove_special_chars:
            # Mantém acentos mas remove pontuação
            text = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', text)
        
        if normalize_whitespace:
            # Normaliza quebras de linha
            text = self.patterns['multiple_newlines'].sub('\n', text)
            # Normaliza espaços
            text = self.patterns['multiple_spaces'].sub(' ', text)
        
        return text.strip()
    
    def normalize_text(self, text: str, 
                      lowercase: bool = True,
                      remove_accents: bool = False) -> str:
        """
        Normaliza o texto.
        
        Args:
            text: Texto a ser normalizado
            lowercase: Converte para minúsculas
            remove_accents: Remove acentos
            
        Returns:
            str: Texto normalizado
        """
        if not text or not isinstance(text, str):
            return ""
        
        if lowercase:
            text = text.lower()
        
        if remove_accents:
            # Remove acentos mantendo caracteres base
            text = unicodedata.normalize('NFD', text)
            text = ''.join(char for char in text 
                          if unicodedata.category(char) != 'Mn')
        
        return text
    
    def tokenize(self, text: str, 
                 method: str = 'word',
                 min_length: int = 2) -> List[str]:
        """
        Tokeniza o texto.
        
        Args:
            text: Texto a ser tokenizado
            method: Método de tokenização ('word' ou 'sentence')
            min_length: Comprimento mínimo dos tokens
            
        Returns:
            List[str]: Lista de tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            if method == 'word':
                tokens = word_tokenize(text, language=self.language)
                # Filtra tokens por comprimento mínimo
                tokens = [token for token in tokens if len(token) >= min_length]
            elif method == 'sentence':
                tokens = sent_tokenize(text, language=self.language)
            else:
                # Tokenização simples por espaços
                tokens = text.split()
                tokens = [token for token in tokens if len(token) >= min_length]
            
            return tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed, using simple split: {str(e)}")
            tokens = text.split()
            return [token for token in tokens if len(token) >= min_length]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords da lista de tokens.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            List[str]: Tokens sem stopwords
        """
        if not tokens:
            return []
        
        return [token for token in tokens 
                if token.lower() not in self.stopwords]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Aplica stemming aos tokens.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            List[str]: Tokens com stemming aplicado
        """
        if not tokens:
            return []
        
        try:
            return [self.stemmer.stem(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Stemming failed: {str(e)}")
            return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Aplica lemmatização aos tokens.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            List[str]: Tokens lemmatizados
        """
        if not tokens:
            return []
        
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Lemmatization failed: {str(e)}")
            return tokens
    
    def preprocess_text(self, text: str,
                       clean: bool = True,
                       normalize: bool = True,
                       tokenize: bool = True,
                       remove_stopwords: bool = True,
                       stem: bool = False,
                       lemmatize: bool = False,
                       min_token_length: int = 2) -> List[str]:
        """
        Pipeline completo de pré-processamento de texto.
        
        Args:
            text: Texto a ser processado
            clean: Aplicar limpeza
            normalize: Aplicar normalização
            tokenize: Aplicar tokenização
            remove_stopwords: Remover stopwords
            stem: Aplicar stemming
            lemmatize: Aplicar lemmatização
            min_token_length: Comprimento mínimo dos tokens
            
        Returns:
            List[str]: Tokens processados
        """
        if not text or not isinstance(text, str):
            return []
        
        processed_text = text
        
        # Limpeza
        if clean:
            processed_text = self.clean_text(processed_text)
        
        # Normalização
        if normalize:
            processed_text = self.normalize_text(processed_text)
        
        # Tokenização
        if tokenize:
            tokens = self.tokenize(processed_text, min_length=min_token_length)
        else:
            tokens = [processed_text]
        
        # Remoção de stopwords
        if remove_stopwords and tokenize:
            tokens = self.remove_stopwords(tokens)
        
        # Stemming
        if stem and tokenize:
            tokens = self.stem_tokens(tokens)
        
        # Lemmatização
        if lemmatize and tokenize:
            tokens = self.lemmatize_tokens(tokens)
        
        return tokens
    
    def preprocess_for_tfidf(self, text: str) -> str:
        """
        Pré-processamento específico para TF-IDF.
        
        Args:
            text: Texto a ser processado
            
        Returns:
            str: Texto processado para TF-IDF
        """
        # Limpeza básica mantendo estrutura
        cleaned = self.clean_text(text, 
                                 remove_special_chars=True,
                                 remove_numbers=False)
        
        # Normalização
        normalized = self.normalize_text(cleaned, lowercase=True)
        
        return normalized
    
    def preprocess_for_embeddings(self, text: str) -> str:
        """
        Pré-processamento específico para embeddings.
        
        Args:
            text: Texto a ser processado
            
        Returns:
            str: Texto processado para embeddings
        """
        # Limpeza mais conservadora para preservar contexto
        cleaned = self.clean_text(text,
                                 remove_special_chars=False,
                                 remove_numbers=False)
        
        # Normalização básica
        normalized = self.normalize_text(cleaned, 
                                       lowercase=True,
                                       remove_accents=False)
        
        return normalized
    
    def get_text_statistics(self, text: str) -> dict:
        """
        Calcula estatísticas básicas do texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            dict: Estatísticas do texto
        """
        if not text or not isinstance(text, str):
            return {}
        
        # Tokenização para estatísticas
        words = self.tokenize(text, method='word')
        sentences = self.tokenize(text, method='sentence')
        
        # Caracteres únicos
        unique_chars = set(text)
        
        # Palavras únicas
        unique_words = set(word.lower() for word in words)
        
        return {
            'total_characters': len(text),
            'total_characters_no_spaces': len(text.replace(' ', '')),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'unique_words': len(unique_words),
            'unique_characters': len(unique_chars),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'vocabulary_richness': len(unique_words) / len(words) if words else 0
        }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extrai palavras-chave do texto baseado em frequência.
        
        Args:
            text: Texto para extração
            top_n: Número de palavras-chave a retornar
            
        Returns:
            List[str]: Lista de palavras-chave
        """
        # Pré-processamento
        tokens = self.preprocess_text(text, 
                                    remove_stopwords=True,
                                    stem=True)
        
        if not tokens:
            return []
        
        # Contagem de frequência
        from collections import Counter
        word_freq = Counter(tokens)
        
        # Retorna as mais frequentes
        return [word for word, freq in word_freq.most_common(top_n)]