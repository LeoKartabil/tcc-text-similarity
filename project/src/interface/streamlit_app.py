
"""
Aplicação principal Streamlit para o Sistema de Comparação de Conteúdo de Textos.
Interface web intuitiva conforme especificado no TCC.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
from typing import Dict, List, Optional, Tuple
import io
import base64

# Imports do sistema
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.algorithms.tfidf_comparator import TFIDFComparator
from src.algorithms.sbert_comparator import SBERTComparator
from src.algorithms.hybrid_comparator import HybridComparator
from src.utils.document_processor import DocumentProcessor
from src.utils.text_preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Sistema de Comparação de Textos - TCC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .algorithm-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class TextComparisonApp:
    """Classe principal da aplicação Streamlit."""
    
    def __init__(self):
        """Inicializa a aplicação."""
        self.document_processor = DocumentProcessor()
        self.text_preprocessor = TextPreprocessor()
        
        # Inicializa comparadores (lazy loading)
        self.tfidf_comparator = None
        self.sbert_comparator = None
        self.hybrid_comparator = None
        
        # Cache de resultados
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []
        
        if 'processed_texts' not in st.session_state:
            st.session_state.processed_texts = {}
    
    def initialize_comparators(self):
        """Inicializa os comparadores com cache."""
        if 'comparators_initialized' not in st.session_state:
            with st.spinner('Inicializando modelos de comparação...'):
                try:
                    self.tfidf_comparator = TFIDFComparator(language='portuguese')
                    self.sbert_comparator = SBERTComparator(
                        model_name='paraphrase-multilingual-MiniLM-L12-v2',
                        language='portuguese'
                    )
                    self.hybrid_comparator = HybridComparator(
                        language='portuguese',
                        sbert_model='paraphrase-multilingual-MiniLM-L12-v2'
                    )
                    st.session_state.comparators_initialized = True
                    st.session_state.tfidf_comparator = self.tfidf_comparator
                    st.session_state.sbert_comparator = self.sbert_comparator
                    st.session_state.hybrid_comparator = self.hybrid_comparator
                    
                except Exception as e:
                    st.error(f"Erro ao inicializar modelos: {str(e)}")
                    st.stop()
        else:
            self.tfidf_comparator = st.session_state.tfidf_comparator
            self.sbert_comparator = st.session_state.sbert_comparator
            self.hybrid_comparator = st.session_state.hybrid_comparator
    
    def render_header(self):
        """Renderiza o cabeçalho da aplicação."""
        st.markdown('<h1 class="main-header">Sistema de Comparação de Conteúdo de Textos</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="algorithm-info">
        <h3>Sobre o Sistema</h3>
        <p>Este sistema implementa <strong>três algoritmos complementares</strong> para comparação textual:</p>
        <ul>
            <li><strong>TF-IDF + Similaridade de Cosseno:</strong> Análise lexical baseada em frequência de termos</li>
            <li><strong>SBERT (Sentence-BERT):</strong> Análise semântica usando embeddings neurais</li>
            <li><strong>Algoritmo Híbrido Adaptativo:</strong> Combina ambas as abordagens com ponderação inteligente</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Renderiza a barra lateral com configurações."""
        st.sidebar.title("⚙️ Configurações")
        
        # Seleção do modo de operação
        mode = st.sidebar.selectbox(
            "Modo de Operação",
            ["Comparação Simples", "Comparação com Documentos", "Análise em Lote", "Análise Técnica Detalhada"],
            help="Escolha o tipo de análise que deseja realizar"
        )
        
        # Configurações dos algoritmos
        st.sidebar.subheader("🔧 Algoritmos")
        
        algorithms = st.sidebar.multiselect(
            "Algoritmos a Executar",
            ["TF-IDF", "SBERT", "Híbrido"],
            default=["Híbrido"],
            help="Selecione quais algoritmos executar"
        )
        
        # Configurações avançadas
        with st.sidebar.expander("Configurações Avançadas"):
            show_detailed_analysis = st.checkbox("Análise Detalhada", value=True)
            show_visualizations = st.checkbox("Visualizações", value=True)
            cache_results = st.checkbox("Cache de Resultados", value=True)
            
            # Configurações TF-IDF
            st.subheader("TF-IDF")
            tfidf_ngram_max = st.slider("N-gram Máximo", 1, 3, 2)
            tfidf_max_features = st.number_input("Max Features", 100, 10000, 5000)
            
            # Configurações SBERT
            st.subheader("SBERT")
            sbert_batch_size = st.slider("Batch Size", 8, 64, 32)
        
        # Informações do sistema
        with st.sidebar.expander("ℹ️ Informações do Sistema"):
            if hasattr(st.session_state, 'comparators_initialized'):
                st.success("✅ Modelos carregados")
                if self.sbert_comparator:
                    model_info = self.sbert_comparator.get_model_info()
                    st.write(f"**Modelo SBERT:** {model_info.get('model_name', 'N/A')}")
                    st.write(f"**Dispositivo:** {model_info.get('device', 'N/A')}")
                    st.write(f"**Dimensão:** {model_info.get('embedding_dimension', 'N/A')}")
            else:
                st.warning("⏳ Modelos não carregados")
        
        return {
            'mode': mode,
            'algorithms': algorithms,
            'show_detailed_analysis': show_detailed_analysis,
            'show_visualizations': show_visualizations,
            'cache_results': cache_results,
            'tfidf_config': {
                'ngram_range': (1, tfidf_ngram_max),
                'max_features': tfidf_max_features
            },
            'sbert_config': {
                'batch_size': sbert_batch_size
            }
        }
    
    def render_text_input_section(self) -> Tuple[str, str]:
        """Renderiza seção de entrada de texto."""
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Entrada de Textos</h2>', unsafe_allow_html=True)
        
        # Opções de entrada
        input_method = st.radio(
            "Método de Entrada",
            ["Texto Direto", "Upload de Arquivos", "Textos de Exemplo"],
            horizontal=True
        )
        
        text1, text2 = "", ""
        
        if input_method == "Texto Direto":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Texto 1")
                text1 = st.text_area(
                    "Digite o primeiro texto:",
                    height=200,
                    placeholder="Cole ou digite o primeiro texto aqui..."
                )
                if text1:
                    st.info(f"📊 {len(text1)} caracteres, {len(text1.split())} palavras")
            
            with col2:
                st.subheader("Texto 2")
                text2 = st.text_area(
                    "Digite o segundo texto:",
                    height=200,
                    placeholder="Cole ou digite o segundo texto aqui..."
                )
                if text2:
                    st.info(f"📊 {len(text2)} caracteres, {len(text2.split())} palavras")
        
        elif input_method == "Upload de Arquivos":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Arquivo 1")
                file1 = st.file_uploader(
                    "Selecione o primeiro arquivo:",
                    type=['txt', 'pdf', 'docx'],
                    key="file1"
                )
                if file1:
                    try:
                        text1 = self.document_processor.extract_text_from_uploaded_file(file1)
                        st.success(f"✅ Arquivo processado: {len(text1)} caracteres")
                        with st.expander("Prévia do texto"):
                            st.text(text1[:500] + "..." if len(text1) > 500 else text1)
                    except Exception as e:
                        st.error(f"Erro ao processar arquivo: {str(e)}")
            
            with col2:
                st.subheader("Arquivo 2")
                file2 = st.file_uploader(
                    "Selecione o segundo arquivo:",
                    type=['txt', 'pdf', 'docx'],
                    key="file2"
                )
                if file2:
                    try:
                        text2 = self.document_processor.extract_text_from_uploaded_file(file2)
                        st.success(f"✅ Arquivo processado: {len(text2)} caracteres")
                        with st.expander("Prévia do texto"):
                            st.text(text2[:500] + "..." if len(text2) > 500 else text2)
                    except Exception as e:
                        st.error(f"Erro ao processar arquivo: {str(e)}")
        
        elif input_method == "Textos de Exemplo":
            example_pair = st.selectbox(
                "Escolha um par de exemplo:",
                [
                    "Algoritmos de Ordenação",
                    "Conceitos de IA",
                    "Textos Similares",
                    "Textos Diferentes"
                ]
            )
            
            examples = self.get_example_texts()
            if example_pair in examples:
                text1, text2 = examples[example_pair]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Texto 1")
                    st.text_area("", value=text1, height=200, disabled=True)
                
                with col2:
                    st.subheader("Texto 2")
                    st.text_area("", value=text2, height=200, disabled=True)
        
        return text1, text2
    
    def get_example_texts(self) -> Dict[str, Tuple[str, str]]:
        """Retorna textos de exemplo para demonstração."""
        return {
            "Algoritmos de Ordenação": (
                """O algoritmo Bubble Sort é um dos métodos de ordenação mais simples de entender e implementar. 
                Ele funciona comparando elementos adjacentes e trocando-os se estiverem na ordem errada. 
                Este processo é repetido até que nenhuma troca seja necessária, indicando que o array está ordenado. 
                Embora seja fácil de compreender, o Bubble Sort tem complexidade O(n²) no pior caso, 
                tornando-o ineficiente para grandes conjuntos de dados.""",
                
                """O Quick Sort é um algoritmo de ordenação eficiente que utiliza a estratégia dividir para conquistar. 
                Ele seleciona um elemento como pivô e particiona o array de forma que elementos menores ficam à esquerda 
                e maiores à direita do pivô. Em seguida, aplica recursivamente o mesmo processo às sublistas. 
                Com complexidade média O(n log n), o Quick Sort é significativamente mais rápido que algoritmos 
                quadráticos como o Bubble Sort para grandes volumes de dados."""
            ),
            
            "Conceitos de IA": (
                """Inteligência Artificial é um campo da ciência da computação que se concentra na criação de 
                sistemas capazes de realizar tarefas que normalmente requerem inteligência humana. 
                Isso inclui aprendizado, raciocínio, percepção e tomada de decisões. 
                A IA utiliza algoritmos complexos e grandes quantidades de dados para treinar modelos 
                que podem reconhecer padrões e fazer previsões.""",
                
                """Machine Learning é uma subárea da inteligência artificial que permite aos computadores 
                aprender e melhorar automaticamente através da experiência, sem serem explicitamente programados. 
                Os algoritmos de ML constroem modelos matemáticos baseados em dados de treinamento 
                para fazer previsões ou decisões. Existem três tipos principais: supervisionado, 
                não supervisionado e por reforço."""
            ),
            
            "Textos Similares": (
                """O desenvolvimento de software é um processo complexo que envolve várias etapas, 
                desde a análise de requisitos até a manutenção do sistema. É fundamental seguir 
                metodologias adequadas e boas práticas para garantir a qualidade do produto final. 
                A documentação e os testes são aspectos cruciais que não devem ser negligenciados.""",
                
                """A engenharia de software abrange todo o ciclo de vida do desenvolvimento de sistemas, 
                incluindo levantamento de requisitos, design, implementação e manutenção. 
                É essencial aplicar metodologias apropriadas e seguir padrões de qualidade 
                para assegurar o sucesso do projeto. Documentação adequada e testes rigorosos 
                são elementos indispensáveis no processo."""
            ),
            
            "Textos Diferentes": (
                """A culinária italiana é famosa mundialmente por sua diversidade e sabores únicos. 
                Pratos como pizza, pasta e risotto conquistaram paladares em todos os continentes. 
                A tradição culinária italiana valoriza ingredientes frescos e técnicas tradicionais 
                passadas de geração em geração. Cada região da Itália possui suas especialidades 
                e características gastronômicas distintas.""",
                
                """A programação orientada a objetos é um paradigma fundamental na engenharia de software. 
                Conceitos como encapsulamento, herança e polimorfismo permitem criar sistemas mais 
                organizados e reutilizáveis. Classes e objetos são os elementos básicos deste paradigma, 
                facilitando a modelagem de problemas do mundo real em código computacional."""
            )
        }
    
    def run_comparison(self, text1: str, text2: str, config: Dict) -> Dict:
        """Executa a comparação entre textos."""
        if not text1.strip() or not text2.strip():
            st.error("⚠️ Ambos os textos devem ser fornecidos!")
            return {}
        
        results = {}
        
        with st.spinner('🔄 Executando comparações...'):
            progress_bar = st.progress(0)
            
            # Placeholders para mensagens de status
            tfidf_placeholder = st.empty()
            sbert_placeholder = st.empty()
            hybrid_placeholder = st.empty()
            
            # TF-IDF
            if "TF-IDF" in config['algorithms']:
                try:
                    progress_bar.progress(0.2)
                    tfidf_placeholder.info("🔍 Executando análise TF-IDF...")
                    
                    # Configura TF-IDF
                    self.tfidf_comparator.ngram_range = config['tfidf_config']['ngram_range']
                    self.tfidf_comparator.max_features = config['tfidf_config']['max_features']
                    
                    results['tfidf'] = self.tfidf_comparator.compare_texts(text1, text2)
                    progress_bar.progress(0.4)
                    tfidf_placeholder.empty()  # Limpa a mensagem após conclusão
                    
                except Exception as e:
                    tfidf_placeholder.empty()
                    st.error(f"Erro no TF-IDF: {str(e)}")
            
            # SBERT
            if "SBERT" in config['algorithms']:
                try:
                    progress_bar.progress(0.5)
                    sbert_placeholder.info("🧠 Executando análise SBERT...")
                    
                    results['sbert'] = self.sbert_comparator.compare_texts(text1, text2)
                    progress_bar.progress(0.7)
                    sbert_placeholder.empty()  # Limpa a mensagem após conclusão
                    
                except Exception as e:
                    sbert_placeholder.empty()
                    st.error(f"Erro no SBERT: {str(e)}")
            
            # Híbrido
            if "Híbrido" in config['algorithms']:
                try:
                    progress_bar.progress(0.8)
                    hybrid_placeholder.info("⚖️ Executando análise híbrida...")
                    
                    results['hybrid'] = self.hybrid_comparator.compare_texts(text1, text2)
                    progress_bar.progress(1.0)
                    hybrid_placeholder.empty()  # Limpa a mensagem após conclusão
                    
                except Exception as e:
                    hybrid_placeholder.empty()
                    st.error(f"Erro no algoritmo híbrido: {str(e)}")
            
            progress_bar.empty()
        
        # Cache dos resultados
        if config['cache_results'] and results:
            st.session_state.comparison_results.append({
                'timestamp': pd.Timestamp.now(),
                'text1_preview': text1[:100] + "..." if len(text1) > 100 else text1,
                'text2_preview': text2[:100] + "..." if len(text2) > 100 else text2,
                'results': results
            })
        
        return results
    
    def render_results(self, results: Dict, config: Dict):
        """Renderiza os resultados da comparação."""
        if not results:
            return
        
        st.markdown('<h2 class="sub-header">📊 Resultados da Comparação</h2>', unsafe_allow_html=True)
        
        # Resumo dos resultados
        self.render_results_summary(results)
        
        # Resultados detalhados por algoritmo
        for algorithm, result in results.items():
            self.render_algorithm_results(algorithm, result, config)
        
        # Visualizações comparativas
        if config['show_visualizations'] and len(results) > 1:
            self.render_comparative_visualizations(results)
        
        # Análise comparativa
        if len(results) > 1:
            self.render_comparative_analysis(results)
    
    def render_results_summary(self, results: Dict):
        """Renderiza resumo dos resultados."""
        st.subheader("📈 Resumo dos Resultados")
        
        cols = st.columns(len(results))
        
        for i, (algorithm, result) in enumerate(results.items()):
            with cols[i]:
                algorithm_name = {
                    'tfidf': 'TF-IDF',
                    'sbert': 'SBERT',
                    'hybrid': 'Híbrido'
                }.get(algorithm, algorithm)
                
                similarity = result['similarity_percentage']
                
                # Cor baseada na similaridade
                if similarity >= 70:
                    color = "🟢"
                elif similarity >= 40:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.metric(
                    label=f"{color} {algorithm_name}",
                    value=f"{similarity:.1f}%",
                    delta=f"{result['similarity_score']:.3f}"
                )
                
                st.caption(result['interpretation'])
    
    def render_algorithm_results(self, algorithm: str, result: Dict, config: Dict):
        """Renderiza resultados de um algoritmo específico."""
        algorithm_names = {
            'tfidf': 'TF-IDF + Similaridade de Cosseno',
            'sbert': 'SBERT (Sentence-BERT)',
            'hybrid': 'Algoritmo Híbrido Adaptativo'
        }
        
        with st.expander(f"🔍 {algorithm_names.get(algorithm, algorithm)} - Análise Detalhada", expanded=True):
            
            # Informações básicas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Similaridade", f"{result['similarity_percentage']:.1f}%")
            
            with col2:
                st.metric("Score", f"{result['similarity_score']:.4f}")
            
            with col3:
                if algorithm == 'hybrid':
                    weights = result.get('weights', {})
                    st.write("**Pesos:**")
                    st.write(f"TF-IDF: {weights.get('tfidf_weight', 0):.1%}")
                    st.write(f"SBERT: {weights.get('sbert_weight', 0):.1%}")
                elif algorithm == 'sbert':
                    st.metric("Dimensão", result.get('embedding_dimension', 'N/A'))
            
            # Interpretação
            st.info(f"**Interpretação:** {result['interpretation']}")
            
            # Análise detalhada
            if config['show_detailed_analysis'] and 'analysis' in result:
                self.render_detailed_analysis(algorithm, result['analysis'])
    
    def render_detailed_analysis(self, algorithm: str, analysis: Dict):
        """Renderiza análise detalhada de um algoritmo."""
        st.subheader("🔬 Análise Detalhada")
        
        if algorithm == 'tfidf':
            self.render_tfidf_analysis(analysis)
        elif algorithm == 'sbert':
            self.render_sbert_analysis(analysis)
        elif algorithm == 'hybrid':
            self.render_hybrid_analysis(analysis)
    
    def render_tfidf_analysis(self, analysis: Dict):
        """Renderiza análise TF-IDF."""
        # Termos mais importantes
        if 'top_terms_text1' in analysis and 'top_terms_text2' in analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Termos - Texto 1:**")
                terms1 = analysis['top_terms_text1'][:5]
                for term, score in terms1:
                    st.write(f"• {term}: {score:.3f}")
            
            with col2:
                st.write("**Top Termos - Texto 2:**")
                terms2 = analysis['top_terms_text2'][:5]
                for term, score in terms2:
                    st.write(f"• {term}: {score:.3f}")
        
        # Termos comuns
        if 'common_terms' in analysis:
            st.write("**Termos Comuns:**")
            common_terms = analysis['common_terms'][:5]
            for term, score1, score2 in common_terms:
                st.write(f"• {term}: {score1:.3f} | {score2:.3f}")
        
        # Diversidade lexical
        if 'lexical_diversity' in analysis:
            diversity = analysis['lexical_diversity']
            st.write("**Diversidade Lexical:**")
            st.write(f"• Similaridade Jaccard: {diversity.get('jaccard_similarity', 0):.3f}")
            st.write(f"• Tokens únicos T1: {diversity.get('unique_tokens_text1', 0)}")
            st.write(f"• Tokens únicos T2: {diversity.get('unique_tokens_text2', 0)}")
    
    def render_sbert_analysis(self, analysis: Dict):
        """Renderiza análise SBERT."""
        # Estatísticas dos embeddings
        if 'embedding_stats' in analysis:
            stats = analysis['embedding_stats']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Embedding Texto 1:**")
                st.write(f"• Média: {stats.get('mean_embedding1', 0):.4f}")
                st.write(f"• Desvio: {stats.get('std_embedding1', 0):.4f}")
            
            with col2:
                st.write("**Embedding Texto 2:**")
                st.write(f"• Média: {stats.get('mean_embedding2', 0):.4f}")
                st.write(f"• Desvio: {stats.get('std_embedding2', 0):.4f}")
        
        # Métricas adicionais
        col1, col2 = st.columns(2)
        
        with col1:
            if 'euclidean_distance' in analysis:
                st.metric("Distância Euclidiana", f"{analysis['euclidean_distance']:.4f}")
        
        with col2:
            if 'pearson_correlation' in analysis:
                st.metric("Correlação Pearson", f"{analysis['pearson_correlation']:.4f}")
        
        # Análise por sentenças
        if 'sentence_similarities' in analysis:
            sent_analysis = analysis['sentence_similarities']
            if 'avg_similarity' in sent_analysis:
                st.write("**Análise por Sentenças:**")
                st.write(f"• Similaridade média: {sent_analysis['avg_similarity']:.3f}")
                st.write(f"• Similaridade máxima: {sent_analysis['max_similarity']:.3f}")
                st.write(f"• Similaridade mínima: {sent_analysis['min_similarity']:.3f}")
    
    def render_hybrid_analysis(self, analysis: Dict):
        """Renderiza análise híbrida."""
        # Justificativa dos pesos
        if 'weight_justification' in analysis:
            st.info(f"**Justificativa dos Pesos:** {analysis['weight_justification']}")
        
        # Características dos textos
        if 'text_characteristics' in analysis:
            chars = analysis['text_characteristics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Características Texto 1:**")
                char1 = chars.get('text1', {})
                st.write(f"• Tipo: {char1.get('text_type', 'N/A')}")
                st.write(f"• Técnico: {'Sim' if char1.get('is_technical', False) else 'Não'}")
                st.write(f"• Complexidade: {char1.get('complexity_score', 0):.2f}")
            
            with col2:
                st.write("**Características Texto 2:**")
                char2 = chars.get('text2', {})
                st.write(f"• Tipo: {char2.get('text_type', 'N/A')}")
                st.write(f"• Técnico: {'Sim' if char2.get('is_technical', False) else 'Não'}")
                st.write(f"• Complexidade: {char2.get('complexity_score', 0):.2f}")
    
    def render_comparative_visualizations(self, results: Dict):
        """Renderiza visualizações comparativas."""
        st.subheader("📊 Visualizações Comparativas")
        
        # Gráfico de barras com similaridades
        algorithms = list(results.keys())
        similarities = [results[alg]['similarity_percentage'] for alg in algorithms]
        
        algorithm_names = {
            'tfidf': 'TF-IDF',
            'sbert': 'SBERT',
            'hybrid': 'Híbrido'
        }
        
        display_names = [algorithm_names.get(alg, alg) for alg in algorithms]
        
        fig = px.bar(
            x=display_names,
            y=similarities,
            title="Comparação de Similaridade por Algoritmo",
            labels={'x': 'Algoritmo', 'y': 'Similaridade (%)'},
            color=similarities,
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico radar se houver híbrido
        if 'hybrid' in results and len(results) >= 2:
            self.render_radar_chart(results)
    
    def render_radar_chart(self, results: Dict):
        """Renderiza gráfico radar para comparação."""
        categories = []
        values = []
        
        algorithm_names = {
            'tfidf': 'TF-IDF',
            'sbert': 'SBERT',
            'hybrid': 'Híbrido'
        }
        
        for alg, result in results.items():
            categories.append(algorithm_names.get(alg, alg))
            values.append(result['similarity_percentage'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Similaridade'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Comparação Radar - Similaridade por Algoritmo"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_comparative_analysis(self, results: Dict):
        """Renderiza análise comparativa entre algoritmos."""
        st.subheader("🔍 Análise Comparativa")
        
        if len(results) < 2:
            return
        
        # Calcula estatísticas comparativas
        similarities = [result['similarity_percentage'] for result in results.values()]
        algorithms = list(results.keys())
        
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        max_diff = max(similarities) - min(similarities)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Similaridade Média", f"{mean_similarity:.1f}%")
        
        with col2:
            st.metric("Desvio Padrão", f"{std_similarity:.1f}%")
        
        with col3:
            st.metric("Diferença Máxima", f"{max_diff:.1f}%")
        
        # Análise de concordância
        if max_diff <= 10:
            st.success("✅ **Alta concordância** entre algoritmos")
        elif max_diff <= 25:
            st.warning("⚠️ **Concordância moderada** entre algoritmos")
        else:
            st.error("❌ **Baixa concordância** entre algoritmos")
        
        # Recomendações
        st.subheader("💡 Recomendações")
        
        if 'hybrid' in results:
            hybrid_result = results['hybrid']
            if 'concordance' in hybrid_result:
                concordance = hybrid_result['concordance']
                st.info(f"**Análise Híbrida:** {concordance['description']}")
                st.write(f"**Método Preferido:** {concordance['preferred_method']}")
                st.write(f"**Nota:** {concordance['confidence_note']}")
        
        # Tabela comparativa
        algorithm_names = {
            'tfidf': 'TF-IDF',
            'sbert': 'SBERT',
            'hybrid': 'Híbrido'
        }
        
        comparison_data = []
        for alg, result in results.items():
            comparison_data.append({
                'Algoritmo': algorithm_names.get(alg, alg),
                'Similaridade (%)': f"{result['similarity_percentage']:.1f}",
                'Score': f"{result['similarity_score']:.4f}",
                'Interpretação': result['interpretation']
            })
        
        df = pd.DataFrame(comparison_data)
        st.table(df)
    
    def render_batch_analysis_mode(self, config: Dict):
        """Renderiza modo de análise em lote."""
        st.markdown('<h2 class="sub-header">📚 Análise em Lote</h2>', unsafe_allow_html=True)
        
        st.info("Este modo permite comparar múltiplos textos simultaneamente.")
        
        # Upload de múltiplos arquivos
        uploaded_files = st.file_uploader(
            "Selecione múltiplos arquivos:",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True
        )
        
        texts = []
        file_names = []
        
        if uploaded_files:
            for file in uploaded_files:
                try:
                    text = self.document_processor.extract_text_from_uploaded_file(file)
                    texts.append(text)
                    file_names.append(file.name)
                except Exception as e:
                    st.error(f"Erro ao processar {file.name}: {str(e)}")
        
        # Entrada manual adicional
        manual_texts = st.text_area(
            "Ou adicione textos manualmente (um por linha):",
            height=150,
            placeholder="Digite cada texto em uma linha separada..."
        )
        
        if manual_texts:
            manual_list = [text.strip() for text in manual_texts.split('\n') if text.strip()]
            texts.extend(manual_list)
            file_names.extend([f"Texto_{i+len(file_names)+1}" for i in range(len(manual_list))])
        
        if len(texts) >= 2:
            st.success(f"✅ {len(texts)} textos carregados para análise")
            
            if st.button("🚀 Executar Análise em Lote"):
                self.run_batch_analysis(texts, file_names, config)
        else:
            st.warning("⚠️ Necessário pelo menos 2 textos para análise em lote")
    
    def run_batch_analysis(self, texts: List[str], names: List[str], config: Dict):
        """Executa análise em lote."""
        with st.spinner('🔄 Executando análise em lote...'):
            
            # Seleciona algoritmo para análise em lote
            if "Híbrido" in config['algorithms']:
                comparator = self.hybrid_comparator
                algorithm_name = "Híbrido"
            elif "SBERT" in config['algorithms']:
                comparator = self.sbert_comparator
                algorithm_name = "SBERT"
            else:
                comparator = self.tfidf_comparator
                algorithm_name = "TF-IDF"
            
            # Executa comparação em lote
            similarity_matrix = comparator.batch_compare(texts)
            
            # Renderiza resultados
            st.subheader(f"📊 Matriz de Similaridade - {algorithm_name}")
            
            # Atualiza índices com nomes dos arquivos
            similarity_matrix.index = names
            similarity_matrix.columns = names
            
            # Heatmap
            fig = px.imshow(
                similarity_matrix.values,
                labels=dict(x="Textos", y="Textos", color="Similaridade"),
                x=names,
                y=names,
                color_continuous_scale='RdYlGn',
                title=f"Matriz de Similaridade - {algorithm_name}"
            )
            
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de similaridades
            st.subheader("📋 Tabela de Similaridades")
            
            # Formata valores para exibição
            display_matrix = similarity_matrix.copy()
            for i in range(len(display_matrix)):
                for j in range(len(display_matrix.columns)):
                    if i != j:  # Não formatar diagonal principal
                        display_matrix.iloc[i, j] = f"{display_matrix.iloc[i, j]:.3f}"
                    else:
                        display_matrix.iloc[i, j] = "1.000"
            
            st.dataframe(display_matrix, use_container_width=True)
            
            # Análise de clusters
            if len(texts) >= 3:
                self.render_cluster_analysis(texts, names, comparator)
    
    def render_cluster_analysis(self, texts: List[str], names: List[str], comparator):
        """Renderiza análise de clusters."""
        st.subheader("🎯 Análise de Clusters")
        
        n_clusters = st.slider("Número de Clusters", 2, min(5, len(texts)-1), 3)
        
        if st.button("🔍 Executar Clustering"):
            try:
                if hasattr(comparator, 'cluster_texts'):
                    cluster_results = comparator.cluster_texts(texts, n_clusters)
                    
                    # Visualiza clusters
                    cluster_data = []
                    colors = px.colors.qualitative.Set1
                    
                    for cluster_id, cluster_info in cluster_results['clusters'].items():
                        for text_info in cluster_info['texts']:
                            cluster_data.append({
                                'Texto': names[text_info['index']],
                                'Cluster': f"Cluster {cluster_id + 1}",
                                'Texto_Preview': text_info['text'],
                                'Distância_Centróide': cluster_info['avg_distance_to_centroid']
                            })
                    
                    df_clusters = pd.DataFrame(cluster_data)
                    
                    # Gráfico de clusters
                    fig = px.scatter(
                        df_clusters,
                        x='Texto',
                        y='Distância_Centróide',
                        color='Cluster',
                        title="Distribuição dos Textos por Cluster",
                        hover_data=['Texto_Preview']
                    )
                    
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela de clusters
                    st.subheader("📊 Composição dos Clusters")
                    for cluster_id, cluster_info in cluster_results['clusters'].items():
                        with st.expander(f"Cluster {cluster_id + 1} ({cluster_info['size']} textos)"):
                            st.write(f"**Texto Representativo:** {cluster_info['representative_text']}")
                            st.write("**Textos do Cluster:**")
                            for text_info in cluster_info['texts']:
                                st.write(f"• {names[text_info['index']]}")
                
                else:
                    st.warning("Clustering não disponível para este algoritmo")
                    
            except Exception as e:
                st.error(f"Erro no clustering: {str(e)}")
    
    def render_history_section(self):
        """Renderiza seção de histórico de comparações."""
        if st.session_state.comparison_results:
            st.subheader("📚 Histórico de Comparações")
            
            # Controles do histórico
            col1, col2 = st.columns([3, 1])
            
            with col1:
                show_last = st.slider("Mostrar últimas", 1, len(st.session_state.comparison_results), 5)
            
            with col2:
                if st.button("🗑️ Limpar Histórico"):
                    st.session_state.comparison_results = []
                    st.rerun()
            
            # Exibe histórico
            recent_results = st.session_state.comparison_results[-show_last:]
            
            for i, entry in enumerate(reversed(recent_results)):
                with st.expander(f"Comparação {len(recent_results)-i} - {entry['timestamp'].strftime('%H:%M:%S')}"):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Texto 1:**")
                        st.text(entry['text1_preview'])
                    
                    with col2:
                        st.write("**Texto 2:**")
                        st.text(entry['text2_preview'])
                    
                    # Resultados resumidos
                    results_summary = []
                    for alg, result in entry['results'].items():
                        algorithm_name = {
                            'tfidf': 'TF-IDF',
                            'sbert': 'SBERT',
                            'hybrid': 'Híbrido'
                        }.get(alg, alg)
                        
                        results_summary.append({
                            'Algoritmo': algorithm_name,
                            'Similaridade': f"{result['similarity_percentage']:.1f}%",
                            'Interpretação': result['interpretation']
                        })
                    
                    if results_summary:
                        st.table(pd.DataFrame(results_summary))
    
    def run(self):
        """Executa a aplicação principal."""
        # Inicializa comparadores
        self.initialize_comparators()
        
        # Renderiza interface
        self.render_header()
        
        # Configurações da sidebar
        config = self.render_sidebar()
        
        # Modo de operação
        if config['mode'] == "Comparação Simples":
            text1, text2 = self.render_text_input_section()
            
            if st.button("🚀 Executar Comparação", type="primary"):
                results = self.run_comparison(text1, text2, config)
                if results:
                    self.render_results(results, config)
        
        elif config['mode'] == "Comparação com Documentos":
            text1, text2 = self.render_text_input_section()
            
            if st.button("🚀 Executar Comparação", type="primary"):
                results = self.run_comparison(text1, text2, config)
                if results:
                    self.render_results(results, config)
        
        elif config['mode'] == "Análise em Lote":
            self.render_batch_analysis_mode(config)
        
        elif config['mode'] == "Análise Técnica Detalhada":
            text1, text2 = self.render_text_input_section()
            
            if st.button("🚀 Executar Análise Detalhada", type="primary"):
                # Força análise detalhada
                config['show_detailed_analysis'] = True
                config['show_visualizations'] = True
                config['algorithms'] = ["TF-IDF", "SBERT", "Híbrido"]
                
                results = self.run_comparison(text1, text2, config)
                if results:
                    self.render_results(results, config)
                    
                    # Explicação detalhada para modo técnico
                    if 'hybrid' in results:
                        st.subheader("📝 Explicação Técnica Detalhada")
                        explanation = self.hybrid_comparator.explain_comparison(text1, text2)
                        st.text(explanation)
        
        # Seção de histórico
        self.render_history_section()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        <p>Sistema de Comparação de Conteúdo de Textos</p>
        <p>Desenvolvido por Leonardo Kartabil para o Trabalho de Conclusão de Curso de Engenharia de Software- 2025</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Função principal para execução da aplicação."""
    try:
        app = TextComparisonApp()
        app.run()
    except Exception as e:
        st.error(f"Erro na aplicação: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()