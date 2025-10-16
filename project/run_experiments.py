"""
Script para executar experimentos com os dados de amostra e coletar métricas
para a seção de resultados e discussões do TCC.
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from algorithms.tfidf_comparator import TFIDFComparator
from algorithms.sbert_comparator import SBERTComparator
from algorithms.hybrid_comparator import HybridComparator
from utils.document_processor import DocumentProcessor

def load_sample_data():
    """Carrega os dados de amostra."""
    processor = DocumentProcessor()
    data_dir = Path("data/samples")
    
    samples = {}
    
    # Algoritmos (arquivos de código)
    algo_dir = data_dir / "algorithms"
    if algo_dir.exists():
        for file in algo_dir.glob("*"):
            # Lê arquivos de código diretamente
            if file.suffix in ['.cs', '.java', '.py', '.js', '.cpp', '.c']:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                samples[f"algorithm_{file.stem}"] = content
            else:
                try:
                    content = processor.extract_text_from_file(str(file))
                    samples[f"algorithm_{file.stem}"] = content
                except Exception as e:
                    print(f"Erro ao processar {file}: {e}")
    
    # Documentos DOCX
    docx_dir = data_dir / "requirements_docx"
    if docx_dir.exists():
        for file in docx_dir.glob("*.docx"):
            try:
                content = processor.extract_text_from_file(str(file))
                samples[f"docx_{file.stem}"] = content
            except Exception as e:
                print(f"Erro ao processar {file}: {e}")
    
    # Documentos PDF
    pdf_dir = data_dir / "requirements_pdf"
    if pdf_dir.exists():
        for file in pdf_dir.glob("*.pdf"):
            try:
                content = processor.extract_text_from_file(str(file))
                samples[f"pdf_{file.stem}"] = content
            except Exception as e:
                print(f"Erro ao processar {file}: {e}")
    
    return samples

def run_algorithm_comparison(samples):
    """Executa comparação entre algoritmos."""
    print("\n=== COMPARAÇÃO DE ALGORITMOS ===")
    
    # Inicializa comparadores
    tfidf = TFIDFComparator(language='portuguese')
    sbert = SBERTComparator(model_name='all-MiniLM-L6-v2', language='portuguese')
    hybrid = HybridComparator(language='portuguese', sbert_model='all-MiniLM-L6-v2')
    
    results = []
    
    # Testa com algoritmos (mesmo algoritmo em linguagens diferentes)
    if 'algorithm_bubble_sort.cs' in samples and 'algorithm_bubble_sort.java' in samples:
        text1 = samples['algorithm_bubble_sort.cs']
        text2 = samples['algorithm_bubble_sort.java']
        
        print("\n--- Comparando Bubble Sort C# vs Java ---")
        
        # TF-IDF
        start_time = time.time()
        tfidf_result = tfidf.compare_texts(text1, text2)
        tfidf_time = time.time() - start_time
        
        # SBERT
        start_time = time.time()
        sbert_result = sbert.compare_texts(text1, text2)
        sbert_time = time.time() - start_time
        
        # Híbrido
        start_time = time.time()
        hybrid_result = hybrid.compare_texts(text1, text2)
        hybrid_time = time.time() - start_time
        
        result = {
            'comparison_type': 'algorithms_same_logic',
            'text1_name': 'Bubble Sort C#',
            'text2_name': 'Bubble Sort Java',
            'text1_length': len(text1),
            'text2_length': len(text2),
            'tfidf_score': tfidf_result['similarity_score'],
            'tfidf_time': tfidf_time,
            'sbert_score': sbert_result['similarity_score'],
            'sbert_time': sbert_time,
            'hybrid_score': hybrid_result['similarity_score'],
            'hybrid_time': hybrid_time,
            'hybrid_weights': hybrid_result['weights'],
            'concordance': hybrid_result['concordance']
        }
        
        results.append(result)
        
        print(f"TF-IDF: {tfidf_result['similarity_percentage']:.1f}% ({tfidf_time:.3f}s)")
        print(f"SBERT: {sbert_result['similarity_percentage']:.1f}% ({sbert_time:.3f}s)")
        print(f"Híbrido: {hybrid_result['similarity_percentage']:.1f}% ({hybrid_time:.3f}s)")
        print(f"Pesos híbrido: TF-IDF={hybrid_result['weights']['tfidf_weight']:.2f}, SBERT={hybrid_result['weights']['sbert_weight']:.2f}")
    
    return results

def run_document_comparison(samples):
    """Executa comparação entre documentos."""
    print("\n=== COMPARAÇÃO DE DOCUMENTOS ===")
    
    # Inicializa comparadores
    tfidf = TFIDFComparator(language='portuguese')
    sbert = SBERTComparator(model_name='all-MiniLM-L6-v2', language='portuguese')
    hybrid = HybridComparator(language='portuguese', sbert_model='all-MiniLM-L6-v2')
    
    results = []
    
    # Compara documentos OCR (mesmo conteúdo em idiomas diferentes)
    ocr_en_key = None
    ocr_pt_key = None
    
    for key in samples.keys():
        if 'OCR_Character_Detection_EN' in key:
            ocr_en_key = key
        elif 'OCR_Deteccao_Caracteres' in key:
            ocr_pt_key = key
    
    if ocr_en_key and ocr_pt_key:
        text1 = samples[ocr_en_key]
        text2 = samples[ocr_pt_key]
        
        print("\n--- Comparando OCR EN vs PT (mesmo conteúdo, idiomas diferentes) ---")
        
        # TF-IDF
        start_time = time.time()
        tfidf_result = tfidf.compare_texts(text1, text2)
        tfidf_time = time.time() - start_time
        
        # SBERT
        start_time = time.time()
        sbert_result = sbert.compare_texts(text1, text2)
        sbert_time = time.time() - start_time
        
        # Híbrido
        start_time = time.time()
        hybrid_result = hybrid.compare_texts(text1, text2)
        hybrid_time = time.time() - start_time
        
        result = {
            'comparison_type': 'same_content_different_languages',
            'text1_name': 'OCR Documentation (EN)',
            'text2_name': 'OCR Documentation (PT)',
            'text1_length': len(text1),
            'text2_length': len(text2),
            'tfidf_score': tfidf_result['similarity_score'],
            'tfidf_time': tfidf_time,
            'sbert_score': sbert_result['similarity_score'],
            'sbert_time': sbert_time,
            'hybrid_score': hybrid_result['similarity_score'],
            'hybrid_time': hybrid_time,
            'hybrid_weights': hybrid_result['weights'],
            'concordance': hybrid_result['concordance']
        }
        
        results.append(result)
        
        print(f"TF-IDF: {tfidf_result['similarity_percentage']:.1f}% ({tfidf_time:.3f}s)")
        print(f"SBERT: {sbert_result['similarity_percentage']:.1f}% ({sbert_time:.3f}s)")
        print(f"Híbrido: {hybrid_result['similarity_percentage']:.1f}% ({hybrid_time:.3f}s)")
        print(f"Pesos híbrido: TF-IDF={hybrid_result['weights']['tfidf_weight']:.2f}, SBERT={hybrid_result['weights']['sbert_weight']:.2f}")
    
    # Compara documentos de gestão financeira (conteúdo similar)
    finance_keys = [key for key in samples.keys() if 'gestao_financeira' in key]
    
    if len(finance_keys) >= 2:
        text1 = samples[finance_keys[0]]
        text2 = samples[finance_keys[1]]
        
        print("\n--- Comparando Documentos de Gestão Financeira (conteúdo similar) ---")
        
        # TF-IDF
        start_time = time.time()
        tfidf_result = tfidf.compare_texts(text1, text2)
        tfidf_time = time.time() - start_time
        
        # SBERT
        start_time = time.time()
        sbert_result = sbert.compare_texts(text1, text2)
        sbert_time = time.time() - start_time
        
        # Híbrido
        start_time = time.time()
        hybrid_result = hybrid.compare_texts(text1, text2)
        hybrid_time = time.time() - start_time
        
        result = {
            'comparison_type': 'similar_domain_documents',
            'text1_name': 'Gestão Financeira Doc 1',
            'text2_name': 'Gestão Financeira Doc 2',
            'text1_length': len(text1),
            'text2_length': len(text2),
            'tfidf_score': tfidf_result['similarity_score'],
            'tfidf_time': tfidf_time,
            'sbert_score': sbert_result['similarity_score'],
            'sbert_time': sbert_time,
            'hybrid_score': hybrid_result['similarity_score'],
            'hybrid_time': hybrid_time,
            'hybrid_weights': hybrid_result['weights'],
            'concordance': hybrid_result['concordance']
        }
        
        results.append(result)
        
        print(f"TF-IDF: {tfidf_result['similarity_percentage']:.1f}% ({tfidf_time:.3f}s)")
        print(f"SBERT: {sbert_result['similarity_percentage']:.1f}% ({sbert_time:.3f}s)")
        print(f"Híbrido: {hybrid_result['similarity_percentage']:.1f}% ({hybrid_time:.3f}s)")
        print(f"Pesos híbrido: TF-IDF={hybrid_result['weights']['tfidf_weight']:.2f}, SBERT={hybrid_result['weights']['sbert_weight']:.2f}")
    
    return results

def run_cross_domain_comparison(samples):
    """Executa comparação entre domínios diferentes."""
    print("\n=== COMPARAÇÃO ENTRE DOMÍNIOS DIFERENTES ===")
    
    # Inicializa comparadores
    tfidf = TFIDFComparator(language='portuguese')
    sbert = SBERTComparator(model_name='all-MiniLM-L6-v2', language='portuguese')
    hybrid = HybridComparator(language='portuguese', sbert_model='all-MiniLM-L6-v2')
    
    results = []
    
    # Compara algoritmo vs documento (domínios muito diferentes)
    algo_key = None
    doc_key = None
    
    for key in samples.keys():
        if 'algorithm_' in key and algo_key is None:
            algo_key = key
        elif 'pdf_' in key and doc_key is None:
            doc_key = key
    
    if algo_key and doc_key:
        text1 = samples[algo_key]
        text2 = samples[doc_key]
        
        print(f"\n--- Comparando {algo_key} vs {doc_key} (domínios diferentes) ---")
        
        # TF-IDF
        start_time = time.time()
        tfidf_result = tfidf.compare_texts(text1, text2)
        tfidf_time = time.time() - start_time
        
        # SBERT
        start_time = time.time()
        sbert_result = sbert.compare_texts(text1, text2)
        sbert_time = time.time() - start_time
        
        # Híbrido
        start_time = time.time()
        hybrid_result = hybrid.compare_texts(text1, text2)
        hybrid_time = time.time() - start_time
        
        result = {
            'comparison_type': 'different_domains',
            'text1_name': algo_key,
            'text2_name': doc_key,
            'text1_length': len(text1),
            'text2_length': len(text2),
            'tfidf_score': tfidf_result['similarity_score'],
            'tfidf_time': tfidf_time,
            'sbert_score': sbert_result['similarity_score'],
            'sbert_time': sbert_time,
            'hybrid_score': hybrid_result['similarity_score'],
            'hybrid_time': hybrid_time,
            'hybrid_weights': hybrid_result['weights'],
            'concordance': hybrid_result['concordance']
        }
        
        results.append(result)
        
        print(f"TF-IDF: {tfidf_result['similarity_percentage']:.1f}% ({tfidf_time:.3f}s)")
        print(f"SBERT: {sbert_result['similarity_percentage']:.1f}% ({sbert_time:.3f}s)")
        print(f"Híbrido: {hybrid_result['similarity_percentage']:.1f}% ({hybrid_time:.3f}s)")
        print(f"Pesos híbrido: TF-IDF={hybrid_result['weights']['tfidf_weight']:.2f}, SBERT={hybrid_result['weights']['sbert_weight']:.2f}")
    
    return results

def analyze_results(all_results):
    """Analisa os resultados coletados."""
    print("\n=== ANÁLISE DOS RESULTADOS ===")
    
    df = pd.DataFrame(all_results)
    
    # Estatísticas gerais
    print("\n--- Estatísticas Gerais ---")
    print(f"Total de comparações: {len(df)}")
    
    # Médias por algoritmo
    print(f"\nSimilaridade média TF-IDF: {df['tfidf_score'].mean():.3f}")
    print(f"Similaridade média SBERT: {df['sbert_score'].mean():.3f}")
    print(f"Similaridade média Híbrido: {df['hybrid_score'].mean():.3f}")
    
    # Tempos de execução
    print(f"\nTempo médio TF-IDF: {df['tfidf_time'].mean():.3f}s")
    print(f"Tempo médio SBERT: {df['sbert_time'].mean():.3f}s")
    print(f"Tempo médio Híbrido: {df['hybrid_time'].mean():.3f}s")
    
    # Análise por tipo de comparação
    print("\n--- Análise por Tipo de Comparação ---")
    for comp_type in df['comparison_type'].unique():
        subset = df[df['comparison_type'] == comp_type]
        print(f"\n{comp_type}:")
        print(f"  TF-IDF: {subset['tfidf_score'].mean():.3f}")
        print(f"  SBERT: {subset['sbert_score'].mean():.3f}")
        print(f"  Híbrido: {subset['hybrid_score'].mean():.3f}")
    
    # Análise de concordância
    print("\n--- Análise de Concordância ---")
    concordance_levels = []
    for result in all_results:
        concordance_levels.append(result['concordance']['level'])
    
    concordance_df = pd.Series(concordance_levels).value_counts()
    print("Distribuição de concordância:")
    for level, count in concordance_df.items():
        print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def main():
    """Executa todos os experimentos."""
    print("🚀 INICIANDO EXPERIMENTOS PARA RESULTADOS DO TCC")
    print("=" * 60)
    
    # Carrega dados de amostra
    print("📁 Carregando dados de amostra...")
    samples = load_sample_data()
    print(f"✅ {len(samples)} arquivos carregados")
    
    for key, content in samples.items():
        print(f"  - {key}: {len(content)} caracteres")
    
    # Executa experimentos
    all_results = []
    
    # Comparação de algoritmos
    algo_results = run_algorithm_comparison(samples)
    all_results.extend(algo_results)
    
    # Comparação de documentos
    doc_results = run_document_comparison(samples)
    all_results.extend(doc_results)
    
    # Comparação entre domínios
    cross_results = run_cross_domain_comparison(samples)
    all_results.extend(cross_results)
    
    # Análise dos resultados
    df = analyze_results(all_results)
    
    # Salva resultados
    results_file = "experiment_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    csv_file = "experiment_results.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"\n✅ Resultados salvos em {results_file} e {csv_file}")
    
    return all_results, df

if __name__ == "__main__":
    results, df = main()