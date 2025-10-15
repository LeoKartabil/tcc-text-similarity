"""
Datasets sintéticos para testes do sistema de comparação textual.
Inclui diferentes tipos de textos com similaridades conhecidas.
"""

from typing import List, Tuple, Dict
import random
import string


class SyntheticDatasets:
    """Gerador de datasets sintéticos para testes."""
    
    def __init__(self, seed: int = 42):
        """
        Inicializa o gerador de datasets.
        
        Args:
            seed: Semente para reprodutibilidade
        """
        random.seed(seed)
        self.seed = seed
    
    def get_identical_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares de textos idênticos.
        
        Returns:
            List[Tuple[str, str, float]]: Lista de (texto1, texto2, similaridade_esperada)
        """
        texts = [
            "Este é um algoritmo de ordenação por inserção muito eficiente.",
            "A estrutura de dados árvore binária permite busca rápida.",
            "O processamento de linguagem natural utiliza técnicas avançadas.",
            "Machine learning é uma subárea da inteligência artificial.",
            "A engenharia de software envolve metodologias de desenvolvimento."
        ]
        
        return [(text, text, 1.0) for text in texts]
    
    def get_high_similarity_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares com alta similaridade (70-90%).
        
        Returns:
            List[Tuple[str, str, float]]: Lista de pares similares
        """
        pairs = [
            (
                "O algoritmo de ordenação quicksort é muito eficiente para grandes datasets.",
                "O quicksort é um algoritmo de ordenação eficiente para datasets grandes.",
                0.85
            ),
            (
                "A inteligência artificial utiliza redes neurais para aprendizado.",
                "Redes neurais são utilizadas na inteligência artificial para aprendizado.",
                0.80
            ),
            (
                "O desenvolvimento de software requer análise de requisitos cuidadosa.",
                "Análise cuidadosa de requisitos é necessária no desenvolvimento de software.",
                0.82
            ),
            (
                "Estruturas de dados como árvores facilitam a organização de informações.",
                "Árvores são estruturas de dados que facilitam organizar informações.",
                0.78
            ),
            (
                "O processamento de texto envolve tokenização e normalização.",
                "Tokenização e normalização são etapas do processamento de texto.",
                0.75
            )
        ]
        
        return pairs
    
    def get_medium_similarity_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares com similaridade média (40-70%).
        
        Returns:
            List[Tuple[str, str, float]]: Lista de pares moderadamente similares
        """
        pairs = [
            (
                "Algoritmos de ordenação são fundamentais na ciência da computação.",
                "A ciência da computação estuda estruturas de dados e algoritmos.",
                0.60
            ),
            (
                "Machine learning utiliza dados para treinar modelos preditivos.",
                "Modelos estatísticos são usados para análise de dados.",
                0.55
            ),
            (
                "A engenharia de software aplica princípios de engenharia ao desenvolvimento.",
                "Desenvolvimento de sistemas requer metodologias estruturadas.",
                0.50
            ),
            (
                "Redes neurais artificiais simulam o funcionamento do cérebro humano.",
                "O cérebro humano processa informações de forma complexa.",
                0.45
            ),
            (
                "Bancos de dados relacionais organizam informações em tabelas.",
                "Sistemas de informação gerenciam dados empresariais.",
                0.42
            )
        ]
        
        return pairs
    
    def get_low_similarity_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares com baixa similaridade (10-40%).
        
        Returns:
            List[Tuple[str, str, float]]: Lista de pares pouco similares
        """
        pairs = [
            (
                "Algoritmos de ordenação são eficientes para organizar dados.",
                "A culinária italiana é famosa por suas massas e molhos.",
                0.05
            ),
            (
                "Redes neurais processam informações de forma paralela.",
                "O futebol brasileiro tem uma rica história de conquistas.",
                0.03
            ),
            (
                "Estruturas de dados facilitam o armazenamento de informações.",
                "A música clássica utiliza instrumentos orquestrais tradicionais.",
                0.02
            ),
            (
                "Machine learning requer grandes volumes de dados de treinamento.",
                "A arquitetura gótica caracteriza-se por suas torres altas.",
                0.01
            ),
            (
                "O desenvolvimento de software segue metodologias ágeis.",
                "A fotografia digital revolucionou a captura de imagens.",
                0.02
            )
        ]
        
        return pairs
    
    def get_technical_vs_casual_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares comparando textos técnicos com casuais sobre o mesmo tema.
        
        Returns:
            List[Tuple[str, str, float]]: Lista de pares técnico vs casual
        """
        pairs = [
            (
                "O algoritmo de ordenação por fusão (merge sort) possui complexidade temporal O(n log n) no pior caso, utilizando a estratégia dividir para conquistar através de chamadas recursivas que dividem o array em subarrays menores até atingir o caso base.",
                "O merge sort é um jeito de organizar uma lista dividindo ela pela metade várias vezes até ficar bem pequena, depois junta tudo de volta em ordem.",
                0.65
            ),
            (
                "A implementação de estruturas de dados do tipo árvore binária de busca permite operações de inserção, remoção e busca com complexidade logarítmica quando a árvore está balanceada, mantendo a propriedade de que elementos menores ficam à esquerda e maiores à direita.",
                "Uma árvore binária é como uma árvore de verdade, mas de cabeça para baixo, onde cada galho tem no máximo dois outros galhos, e você organiza os números de um jeito especial para encontrar eles mais rápido.",
                0.58
            ),
            (
                "O paradigma de programação orientada a objetos encapsula dados e comportamentos em entidades denominadas objetos, implementando conceitos como herança, polimorfismo e abstração para promover reutilização de código e manutenibilidade.",
                "Programação orientada a objetos é quando você programa pensando em coisas do mundo real, tipo um carro que tem cor e pode acelerar, e você pode fazer vários carros diferentes usando a mesma receita básica.",
                0.62
            )
        ]
        
        return pairs
    
    def get_multilingual_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares em diferentes idiomas (português/inglês).
        
        Returns:
            List[Tuple[str, str, float]]: Lista de pares multilíngues
        """
        pairs = [
            (
                "Algoritmos de ordenação são fundamentais na ciência da computação.",
                "Sorting algorithms are fundamental in computer science.",
                0.70
            ),
            (
                "Machine learning utiliza dados para treinar modelos preditivos.",
                "Machine learning uses data to train predictive models.",
                0.75
            ),
            (
                "A inteligência artificial revoluciona diversas áreas do conhecimento.",
                "Artificial intelligence revolutionizes various areas of knowledge.",
                0.72
            ),
            (
                "Estruturas de dados organizam informações de forma eficiente.",
                "Data structures organize information efficiently.",
                0.68
            )
        ]
        
        return pairs
    
    def get_code_vs_description_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares comparando código com sua descrição.
        
        Returns:
            List[Tuple[str, str, float]]: Lista de pares código vs descrição
        """
        pairs = [
            (
                """
                def bubble_sort(arr):
                    n = len(arr)
                    for i in range(n):
                        for j in range(0, n-i-1):
                            if arr[j] > arr[j+1]:
                                arr[j], arr[j+1] = arr[j+1], arr[j]
                    return arr
                """,
                "O bubble sort compara elementos adjacentes e os troca se estiverem na ordem errada, repetindo este processo até que nenhuma troca seja necessária, resultando em um array ordenado.",
                0.45
            ),
            (
                """
                class BinaryTree:
                    def __init__(self, value):
                        self.value = value
                        self.left = None
                        self.right = None
                    
                    def insert(self, value):
                        if value < self.value:
                            if self.left is None:
                                self.left = BinaryTree(value)
                            else:
                                self.left.insert(value)
                        else:
                            if self.right is None:
                                self.right = BinaryTree(value)
                            else:
                                self.right.insert(value)
                """,
                "Uma árvore binária é uma estrutura de dados onde cada nó tem no máximo dois filhos, sendo que valores menores são inseridos à esquerda e valores maiores à direita, mantendo a propriedade de busca binária.",
                0.40
            )
        ]
        
        return pairs
    
    def get_paraphrase_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares de paráfrases (mesmo significado, palavras diferentes).
        
        Returns:
            List[Tuple[str, str, float]]: Lista de paráfrases
        """
        pairs = [
            (
                "O algoritmo é muito eficiente para processar grandes quantidades de dados.",
                "Este método computacional demonstra alta performance ao lidar com volumes extensos de informações.",
                0.80
            ),
            (
                "A implementação da estrutura de dados foi bem-sucedida.",
                "O desenvolvimento da organização de informações obteve êxito.",
                0.75
            ),
            (
                "O sistema de machine learning apresentou resultados precisos.",
                "A solução de aprendizado automático demonstrou alta acurácia.",
                0.82
            ),
            (
                "A análise de complexidade temporal é fundamental para otimização.",
                "O estudo da eficiência computacional é essencial para melhorar performance.",
                0.78
            )
        ]
        
        return pairs
    
    def get_negation_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retorna pares com negações (significados opostos).
        
        Returns:
            List[Tuple[str, str, float]]: Lista de pares com negações
        """
        pairs = [
            (
                "Este algoritmo é muito eficiente para grandes datasets.",
                "Este algoritmo não é eficiente para grandes datasets.",
                0.85  # Alta similaridade lexical, mas significados opostos
            ),
            (
                "A implementação está funcionando corretamente.",
                "A implementação não está funcionando corretamente.",
                0.80
            ),
            (
                "O sistema de machine learning é preciso.",
                "O sistema de machine learning é impreciso.",
                0.75
            ),
            (
                "A estrutura de dados permite busca rápida.",
                "A estrutura de dados impede busca rápida.",
                0.70
            )
        ]
        
        return pairs
    
    def generate_random_text_pairs(self, num_pairs: int = 50) -> List[Tuple[str, str, float]]:
        """
        Gera pares de textos aleatórios para testes de robustez.
        
        Args:
            num_pairs: Número de pares a gerar
            
        Returns:
            List[Tuple[str, str, float]]: Lista de pares aleatórios
        """
        words_pool = [
            "algoritmo", "estrutura", "dados", "sistema", "processamento",
            "informação", "computação", "software", "desenvolvimento", "análise",
            "implementação", "otimização", "performance", "eficiência", "método",
            "técnica", "solução", "problema", "resultado", "processo",
            "machine", "learning", "inteligência", "artificial", "neural",
            "rede", "modelo", "treinamento", "predição", "classificação"
        ]
        
        pairs = []
        
        for _ in range(num_pairs):
            # Gera dois textos aleatórios
            text1_words = random.sample(words_pool, random.randint(5, 15))
            text2_words = random.sample(words_pool, random.randint(5, 15))
            
            text1 = " ".join(text1_words)
            text2 = " ".join(text2_words)
            
            # Calcula similaridade aproximada baseada em palavras comuns
            common_words = set(text1_words) & set(text2_words)
            total_words = set(text1_words) | set(text2_words)
            
            similarity = len(common_words) / len(total_words) if total_words else 0
            
            pairs.append((text1, text2, similarity))
        
        return pairs
    
    def get_all_test_datasets(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Retorna todos os datasets de teste organizados por categoria.
        
        Returns:
            Dict[str, List[Tuple[str, str, float]]]: Datasets organizados
        """
        return {
            'identical': self.get_identical_pairs(),
            'high_similarity': self.get_high_similarity_pairs(),
            'medium_similarity': self.get_medium_similarity_pairs(),
            'low_similarity': self.get_low_similarity_pairs(),
            'technical_vs_casual': self.get_technical_vs_casual_pairs(),
            'multilingual': self.get_multilingual_pairs(),
            'code_vs_description': self.get_code_vs_description_pairs(),
            'paraphrases': self.get_paraphrase_pairs(),
            'negations': self.get_negation_pairs(),
            'random': self.generate_random_text_pairs(20)
        }
    
    def get_validation_dataset(self) -> List[Tuple[str, str, float]]:
        """
        Retorna dataset balanceado para validação dos algoritmos.
        
        Returns:
            List[Tuple[str, str, float]]: Dataset de validação
        """
        validation_data = []
        
        # Adiciona amostras de cada categoria
        validation_data.extend(self.get_identical_pairs()[:2])
        validation_data.extend(self.get_high_similarity_pairs()[:3])
        validation_data.extend(self.get_medium_similarity_pairs()[:3])
        validation_data.extend(self.get_low_similarity_pairs()[:2])
        validation_data.extend(self.get_paraphrase_pairs()[:2])
        
        return validation_data
    
    def get_benchmark_dataset(self) -> List[Tuple[str, str, float]]:
        """
        Retorna dataset para benchmark de performance.
        
        Returns:
            List[Tuple[str, str, float]]: Dataset de benchmark
        """
        benchmark_data = []
        
        # Textos de diferentes tamanhos para testar performance
        short_texts = [
            ("algoritmo", "estrutura", 0.0),
            ("machine learning", "aprendizado máquina", 0.8),
            ("dados estruturados", "informações organizadas", 0.7)
        ]
        
        medium_texts = [
            (
                "O algoritmo de ordenação quicksort utiliza divisão e conquista",
                "Quicksort é um método de ordenação baseado em dividir para conquistar",
                0.75
            ),
            (
                "Estruturas de dados como árvores binárias facilitam buscas eficientes",
                "Árvores binárias são estruturas que otimizam operações de busca",
                0.70
            )
        ]
        
        long_texts = [
            (
                "A inteligência artificial é um campo da ciência da computação que se concentra na criação de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana, incluindo aprendizado, raciocínio, percepção e tomada de decisões.",
                "Inteligência artificial representa uma área da computação focada no desenvolvimento de sistemas que podem executar atividades típicas da cognição humana, como aprender, analisar, perceber e decidir.",
                0.80
            )
        ]
        
        benchmark_data.extend(short_texts)
        benchmark_data.extend(medium_texts)
        benchmark_data.extend(long_texts)
        
        return benchmark_data


# Instância global para facilitar importação
synthetic_data = SyntheticDatasets()


def get_test_datasets():
    """Função de conveniência para obter todos os datasets."""
    return synthetic_data.get_all_test_datasets()


def get_validation_data():
    """Função de conveniência para obter dados de validação."""
    return synthetic_data.get_validation_dataset()


def get_benchmark_data():
    """Função de conveniência para obter dados de benchmark."""
    return synthetic_data.get_benchmark_dataset()