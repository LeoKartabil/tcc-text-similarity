"""
Módulo de testes para o Sistema de Comparação de Conteúdo de Textos.
Inclui testes unitários, de integração e datasets de teste.
"""

import sys
from pathlib import Path

# Adiciona o diretório src ao path para importações
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))