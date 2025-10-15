# 🚀 Guia de Instalação e Uso - Sistema de Comparação de Textos

## 📋 Pré-requisitos

### Sistema Operacional
- Windows 10/11, macOS 10.14+, ou Linux Ubuntu 18.04+
- Python 3.9 ou superior
- 4GB RAM mínimo (8GB recomendado)
- 2GB espaço livre em disco

### Verificação do Python
```bash
python --version
# ou
python3 --version
```

Se não tiver Python instalado, baixe em: https://python.org/downloads/


## ▶️ Execução

### Interface Web (Recomendado)
```bash
# Execute a aplicação Streamlit
streamlit run app.py

```

### Teste do Sistema
```bash
# Execute o teste completo
python test_system.py

# Execute testes unitários
pytest tests/

# Execute com cobertura
pytest --cov=src --cov-report=html
```

## 🎯 Como Usar a Interface Web

### 1. Comparação Simples
1. Acesse http://localhost:8501
2. Selecione "Comparação Simples" na barra lateral
3. Digite ou cole dois textos nas caixas
4. Clique em "Executar Comparação"
5. Visualize os resultados com métricas detalhadas

### 2. Upload de Documentos
1. Selecione "Comparação com Documentos"
2. Use "Upload de Arquivos" como método de entrada
3. Carregue arquivos PDF, DOCX ou TXT
4. Execute a comparação

### 3. Análise em Lote
1. Selecione "Análise em Lote"
2. Carregue múltiplos arquivos
3. Visualize matriz de similaridade
4. Explore clusters automáticos

### 4. Configurações Avançadas
- **Algoritmos**: Escolha TF-IDF, SBERT ou Híbrido
- **Parâmetros TF-IDF**: Ajuste n-gramas e features
- **Modelo SBERT**: Selecione modelo apropriado
- **Visualizações**: Ative/desative gráficos

## 💻 Uso Programático

### Exemplo Básico
```python
import sys
sys.path.append('src')

from algorithms.hybrid_comparator import HybridComparator

# Inicializa comparador
comparator = HybridComparator(language='portuguese')

# Compara textos
text1 = "Este é o primeiro texto"
text2 = "Este é o segundo texto"

result = comparator.compare_texts(text1, text2)
print(f"Similaridade: {result['similarity_percentage']:.1f}%")
```

### Exemplo com Documentos
```python
from utils.document_processor import DocumentProcessor
from algorithms.tfidf_comparator import TFIDFComparator

# Processa documentos
processor = DocumentProcessor()
text1 = processor.extract_text_from_file("documento1.pdf")
text2 = processor.extract_text_from_file("documento2.docx")

# Compara usando TF-IDF
comparator = TFIDFComparator()
result = comparator.compare_texts(text1, text2)
```

### Exemplo Análise em Lote
```python
from algorithms.sbert_comparator import SBERTComparator

comparator = SBERTComparator()

texts = [
    "Primeiro texto para comparação",
    "Segundo texto do conjunto",
    "Terceiro texto diferente"
]

# Matriz de similaridade
similarity_matrix = comparator.batch_compare(texts)
print(similarity_matrix)
```

## 🔧 Solução de Problemas

### Erro: "ModuleNotFoundError"
```bash
# Verifique se está no ambiente virtual
# Reinstale dependências
pip install -r requirements.txt
```

### Erro: "NLTK Data not found"
```bash
# Baixe recursos NLTK
python -c "import nltk; nltk.download('all')"
```

### Erro: "CUDA out of memory" (SBERT)
```python
# Use CPU em vez de GPU
comparator = SBERTComparator(device='cpu')
```

### Erro: "Streamlit command not found"
```bash
# Instale Streamlit explicitamente
pip install streamlit
```

### Performance Lenta
1. **Use modelos menores**:
   ```python
   # Modelo mais rápido
   comparator = SBERTComparator(model_name='all-MiniLM-L6-v2')
   ```

3. **Processe em lotes menores**:
   ```python
   # Reduza batch_size se necessário
   # Também é possível alterar o batch_size através da interface
   embeddings = comparator.encode_texts(texts, batch_size=16)
   ```

## 📊 Interpretação dos Resultados

### Scores de Similaridade
- **90-100%**: Textos muito similares ou idênticos
- **70-89%**: Textos similares (paráfrases, mesmo tema)
- **50-69%**: Similaridade moderada (temas relacionados)
- **30-49%**: Baixa similaridade (poucos elementos comuns)
- **0-29%**: Textos muito diferentes

### Algoritmos
- **TF-IDF**: Melhor para textos técnicos e longos
- **SBERT**: Melhor para análise semântica e textos curtos
- **Híbrido**: Combina ambos automaticamente

### Métricas Avançadas
- **Concordância**: Indica confiabilidade do resultado
- **Pesos Adaptativos**: Mostra como o híbrido balanceou os métodos
- **Análise Lexical**: Termos mais importantes identificados

## 🆘 Suporte

### Logs de Debug
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Ativa logs detalhados
```

### Informações do Sistema
```python
# Verifique configurações
print(comparator.get_model_info())
print(comparator.get_statistics())
```

### Contato
- **Desenvolvedor**: Leonardo Kartabil
- **Email**: leonardo.kar@hotmail.com
- **TCC**: Sistema de Comparação de Conteúdos de Textos

---