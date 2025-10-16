"""
Arquivo principal para executar o Sistema de Comparação de Conteúdo de Textos.
Execute com: streamlit run app.py
"""

import sys
from pathlib import Path

# Adiciona o diretório src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importa e executa a aplicação
from interface.streamlit_app import main

if __name__ == "__main__":
    main()