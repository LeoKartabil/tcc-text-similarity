"""
Testes unitários para o módulo DocumentProcessor.
"""

import pytest
import tempfile
import os
from pathlib import Path
import io

from utils.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Testes para a classe DocumentProcessor."""
    
    def setup_method(self):
        """Setup executado antes de cada teste."""
        self.processor = DocumentProcessor()
    
    def test_init(self):
        """Testa inicialização do processador."""
        assert self.processor is not None
        assert self.processor.processed_files == []
        assert self.processor.SUPPORTED_FORMATS == {'.pdf', '.docx', '.txt'}
    
    def test_supported_formats(self):
        """Testa formatos suportados."""
        supported = self.processor.SUPPORTED_FORMATS
        assert '.pdf' in supported
        assert '.docx' in supported
        assert '.txt' in supported
        assert '.jpg' not in supported
    
    def test_extract_from_txt_simple(self):
        """Testa extração de texto simples de arquivo TXT."""
        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            test_content = "Este é um teste de extração de texto.\nSegunda linha do teste."
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Testa extração
            extracted_text = self.processor.extract_text_from_file(temp_path)
            assert extracted_text == test_content
        finally:
            # Limpa arquivo temporário
            os.unlink(temp_path)
    
    def test_extract_from_txt_encoding(self):
        """Testa extração com diferentes encodings."""
        # Testa UTF-8
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            test_content = "Texto com acentos: ção, ã, é, ü"
            f.write(test_content)
            temp_path = f.name
        
        try:
            extracted_text = self.processor.extract_text_from_file(temp_path)
            assert "ção" in extracted_text
            assert "acentos" in extracted_text
        finally:
            os.unlink(temp_path)
    
    def test_extract_from_txt_bytes(self):
        """Testa extração de bytes de arquivo TXT."""
        test_content = "Teste de extração de bytes"
        test_bytes = test_content.encode('utf-8')
        
        extracted_text = self.processor._extract_from_txt_bytes(test_bytes)
        assert extracted_text == test_content
    
    def test_get_file_info_existing(self):
        """Testa obtenção de informações de arquivo existente."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            info = self.processor.get_file_info(temp_path)
            
            assert info['name'] == Path(temp_path).name
            assert info['extension'] == '.txt'
            assert info['is_supported'] is True
            assert info['size'] > 0
            assert 'modified_time' in info
        finally:
            os.unlink(temp_path)
    
    def test_get_file_info_nonexistent(self):
        """Testa obtenção de informações de arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            self.processor.get_file_info("arquivo_inexistente.txt")
    
    def test_validate_file_supported(self):
        """Testa validação de arquivo suportado."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            assert self.processor.validate_file(temp_path) is True
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_unsupported(self):
        """Testa validação de arquivo não suportado."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        
        try:
            assert self.processor.validate_file(temp_path) is False
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_nonexistent(self):
        """Testa validação de arquivo inexistente."""
        assert self.processor.validate_file("arquivo_inexistente.txt") is False
    
    def test_extract_unsupported_format(self):
        """Testa extração de formato não suportado."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Formato não suportado"):
                self.processor.extract_text_from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_extract_nonexistent_file(self):
        """Testa extração de arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            self.processor.extract_text_from_file("arquivo_inexistente.txt")
    
    def test_batch_process_success(self):
        """Testa processamento em lote com sucesso."""
        # Cria múltiplos arquivos temporários
        temp_files = []
        test_contents = ["Conteúdo 1", "Conteúdo 2", "Conteúdo 3"]
        
        for i, content in enumerate(test_contents):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_files.append(f.name)
        
        try:
            # Executa processamento em lote
            results = self.processor.batch_process(temp_files)
            
            assert results['total_processed'] == 3
            assert len(results['successful']) == 3
            assert len(results['failed']) == 0
            
            # Verifica conteúdos
            for i, file_path in enumerate(temp_files):
                assert results['successful'][file_path] == test_contents[i]
        
        finally:
            # Limpa arquivos temporários
            for temp_file in temp_files:
                os.unlink(temp_file)
    
    def test_batch_process_mixed_results(self):
        """Testa processamento em lote com sucessos e falhas."""
        # Cria arquivos válidos e inválidos
        temp_files = []
        
        # Arquivo válido
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Conteúdo válido")
            temp_files.append(f.name)
        
        # Arquivo inexistente
        temp_files.append("arquivo_inexistente.txt")
        
        # Arquivo com formato não suportado
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_files.append(f.name)
        
        try:
            results = self.processor.batch_process(temp_files)
            
            assert results['total_processed'] == 1  # Apenas o arquivo válido
            assert len(results['successful']) == 1
            assert len(results['failed']) == 2
            
            # Verifica arquivo válido
            valid_file = temp_files[0]
            assert results['successful'][valid_file] == "Conteúdo válido"
            
            # Verifica falhas
            assert "arquivo_inexistente.txt" in results['failed']
            assert temp_files[2] in results['failed']
        
        finally:
            # Limpa arquivos temporários (apenas os que existem)
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def test_extract_from_uploaded_file_txt(self):
        """Testa extração de arquivo carregado (simulado)."""
        # Simula arquivo carregado
        class MockUploadedFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content.encode('utf-8')
            
            def read(self):
                return self.content
        
        mock_file = MockUploadedFile("test.txt", "Conteúdo do arquivo carregado")
        
        extracted_text = self.processor.extract_text_from_uploaded_file(mock_file)
        assert extracted_text == "Conteúdo do arquivo carregado"
    
    def test_extract_from_uploaded_file_unsupported(self):
        """Testa extração de arquivo carregado com formato não suportado."""
        class MockUploadedFile:
            def __init__(self, name):
                self.name = name
            
            def read(self):
                return b"fake content"
        
        mock_file = MockUploadedFile("test.jpg")
        
        with pytest.raises(ValueError, match="Formato não suportado"):
            self.processor.extract_text_from_uploaded_file(mock_file)
    
    def test_empty_text_handling(self):
        """Testa tratamento de texto vazio."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("")  # Arquivo vazio
            temp_path = f.name
        
        try:
            extracted_text = self.processor.extract_text_from_file(temp_path)
            assert extracted_text == ""
        finally:
            os.unlink(temp_path)
    
    def test_large_text_handling(self):
        """Testa tratamento de texto grande."""
        # Cria texto grande (10KB)
        large_content = "A" * 10000
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            extracted_text = self.processor.extract_text_from_file(temp_path)
            assert len(extracted_text) == 10000
            assert extracted_text == large_content
        finally:
            os.unlink(temp_path)
    
    def test_special_characters_handling(self):
        """Testa tratamento de caracteres especiais."""
        special_content = "Texto com símbolos: @#$%^&*()_+{}|:<>?[]\\;'\",./"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(special_content)
            temp_path = f.name
        
        try:
            extracted_text = self.processor.extract_text_from_file(temp_path)
            assert extracted_text == special_content
        finally:
            os.unlink(temp_path)


class TestDocumentProcessorIntegration:
    """Testes de integração para DocumentProcessor."""
    
    def setup_method(self):
        """Setup executado antes de cada teste."""
        self.processor = DocumentProcessor()
    
    def test_multiple_file_types_batch(self):
        """Testa processamento em lote com múltiplos tipos de arquivo."""
        temp_files = []
        expected_contents = []
        
        # Arquivo TXT
        txt_content = "Conteúdo do arquivo TXT"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(txt_content)
            temp_files.append(f.name)
            expected_contents.append(txt_content)
        
        try:
            results = self.processor.batch_process(temp_files)
            
            assert results['total_processed'] == 1
            assert len(results['successful']) == 1
            assert results['successful'][temp_files[0]] == expected_contents[0]
        
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def test_error_recovery(self):
        """Testa recuperação de erros durante processamento."""
        # Cria cenário com arquivo corrompido (permissões)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Remove permissões de leitura (se possível no sistema)
            try:
                os.chmod(temp_path, 0o000)
                
                # Tenta processar arquivo sem permissões
                results = self.processor.batch_process([temp_path])
                
                # Deve falhar graciosamente
                assert results['total_processed'] == 0
                assert len(results['failed']) == 1
                assert temp_path in results['failed']
            
            except (OSError, PermissionError):
                # Sistema não suporta mudança de permissões, pula teste
                pytest.skip("Sistema não suporta mudança de permissões")
        
        finally:
            # Restaura permissões e remove arquivo
            try:
                os.chmod(temp_path, 0o644)
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass


if __name__ == "__main__":
    pytest.main([__file__])