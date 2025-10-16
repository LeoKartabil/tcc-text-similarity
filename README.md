# Sistema de Comparação de Conteúdo de Textos

## 📊 Sobre o Projeto

Este sistema implementa **três algoritmos complementares** para comparação textual, desenvolvido como parte do TCC "Sistema de Comparação de Conteúdos de Textos Utilizando Técnicas de Análise Lexical e Semântica" por Leonardo Kartabil.

### 🎯 Algoritmos Implementados

1. **TF-IDF + Similaridade de Cosseno**: Análise lexical baseada em frequência de termos
2. **SBERT (Sentence-BERT)**: Análise semântica usando embeddings neurais
3. **Algoritmo Híbrido Adaptativo**: Combina ambas as abordagens com ponderação inteligente

### ✨ Características Principais

- 📄 **Processamento de Documentos**: Suporte a PDF, DOCX e TXT
- 🌐 **Interface Web Intuitiva**: Desenvolvida em Streamlit
- 🔍 **Análise Detalhada**: Métricas completas e visualizações
- 🧪 **Testes Abrangentes**: Cobertura de testes > 80%
- 🚀 **Performance Otimizada**: Cache de embeddings e processamento eficiente

## 🚀 Instalação e Execução

### Pré-requisitos

- Python 3.10+
- pip ou conda

## 🔧 Instalação

### Passo 1: Clone ou Baixe o Projeto
```bash
# Se usando Git
git clone <repository-url>
cd project

# Ou extraia o arquivo ZIP baixado
```

### Passo 2: Crie um Ambiente Virtual (Recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instale as Dependências
```bash
# Instalação básica
pip install -r requirements.txt
```

### Passo 4: Baixar Recursos NLTK (Primeira Execução)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

A aplicação deve abrir automaticamente no navegor
URL: http://localhost:8501

## 📖 Como Usar

### 1. Interface Web

1. **Comparação Simples**: Digite ou cole dois textos
2. **Upload de Documentos**: Carregue arquivos PDF, DOCX ou TXT
3. **Análise em Lote**: Compare múltiplos textos simultaneamente
4. **Análise Técnica**: Visualize métricas detalhadas

### 2. API Programática

```python
from src.algorithms import TFIDFComparator, SBERTComparator, HybridComparator

# TF-IDF
tfidf = TFIDFComparator(language='portuguese')
result = tfidf.compare_texts(text1, text2)

# SBERT
sbert = SBERTComparator(model_name='paraphrase-multilingual-MiniLM-L12-v2')
result = sbert.compare_texts(text1, text2)

# Híbrido
hybrid = HybridComparator(language='portuguese')
result = hybrid.compare_texts(text1, text2)

print(f"Similaridade: {result['similarity_percentage']:.1f}%")
```

### 3. Processamento de Documentos

```python
from src.utils import DocumentProcessor

processor = DocumentProcessor()

# Extrair texto de arquivo
text = processor.extract_text_from_file("documento.pdf")

# Processamento em lote
results = processor.batch_process(["doc1.pdf", "doc2.docx", "doc3.txt"])
```

## 🧪 Testes

```bash
# Execute todos os testes
pytest

# Testes com cobertura
pytest --cov=src --cov-report=html

# Testes específicos
pytest tests/unit/test_tfidf_comparator.py
pytest tests/unit/test_document_processor.py
```

## 📊 Estrutura do Projeto

```
project/
├── src/                          # Código fonte
│   ├── algorithms/              # Algoritmos de comparação
│   │   ├── tfidf_comparator.py  # TF-IDF + Cosseno
│   │   ├── sbert_comparator.py  # SBERT
│   │   └── hybrid_comparator.py # Híbrido Adaptativo
│   ├── utils/                   # Utilitários
│   │   ├── document_processor.py # Processamento de docs
│   │   └── text_preprocessor.py  # Pré-processamento
│   └── interface/               # Interface web
│       └── streamlit_app.py     # App Streamlit
├── tests/                       # Testes
│   ├── unit/                    # Testes unitários
│   └── datasets/                # Datasets de teste
├── data/                        # Dados e amostras
│   └── samples/                 # Arquivos de exemplo para testes
├── app.py                       # Arquivo principal
├── run_experiments.py           # Script de experimentos
├── setup.py                     # Configuração do pacote
├── pytest.ini                   # Configuração de testes
├── requirements.txt             # Dependências
├── TECH_DOCUMENTATION.md       # Documentação técnica
└── README.md                    # Este arquivo
```

## 🔧 Configuração Avançada

### Modelos SBERT

O sistema suporta diferentes modelos SBERT:

- `paraphrase-multilingual-MiniLM-L12-v2` (padrão, multilíngue)
- `all-MiniLM-L6-v2` (inglês, rápido)
- `all-mpnet-base-v2` (inglês, preciso)

### Configurações TF-IDF

```python
comparator = TFIDFComparator(
    language='portuguese',
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)
```

### Algoritmo Híbrido

```python
hybrid = HybridComparator(
    language='portuguese',
    default_tfidf_weight=0.4,
    default_sbert_weight=0.6,
    adaptive_weighting=True  # Ponderação automática
)
```

## 👨‍💻 Autor

**Leonardo Kartabil**
- Email: leonardo.kar@hotmail.com
- TCC: "Sistema de Comparação de Conteúdos de Textos Utilizando Técnicas de Análise Lexical e Semântica"

## 🙏 Agradecimentos

- Prof. Allan Christian Krainski Ferrari (Orientador)
- Centro Universitário Internacional (UNINTER)
- Comunidade open source pelas bibliotecas utilizadas

## 📚 Referências
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Gomaa, W. H., & Fahmy, A. A. (2013). A Survey of Text Similarity Approaches
- Shahmirzadi, O., Lugowski, A., & Younge, K. (2019). Text Similarity in Vector Space Models: A Comparative Study

---