import pytest
import asyncio
from unittest.mock import Mock, patch
from azure_pdf_processor import AzurePDFProcessor
from config import AzureConfig

@pytest.fixture
def mock_config():
    return {
        "storage_account_name": "test_storage",
        "storage_account_key": "test_key",
        "doc_intelligence_endpoint": "https://test.cognitiveservices.azure.com/",
        "doc_intelligence_key": "test_key",
        "computer_vision_endpoint": "https://test.cognitiveservices.azure.com/",
        "computer_vision_key": "test_key",
        "search_endpoint": "https://test.search.windows.net",
        "search_key": "test_key",
        "search_index_name": "test-index",
        "openai_endpoint": "https://test.openai.azure.com/",
        "openai_key": "test_key"
    }

@pytest.fixture
def processor(mock_config):
    with patch('azure_pdf_processor.BlobServiceClient'), \
         patch('azure_pdf_processor.DocumentAnalysisClient'), \
         patch('azure_pdf_processor.SearchIndexClient'), \
         patch('azure_pdf_processor.SearchClient'):
        return AzurePDFProcessor(mock_config)

@pytest.mark.asyncio
async def test_get_embedding(processor):
    with patch('openai.Embedding.create') as mock_openai:
        mock_openai.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
        }
        
        result = await processor.get_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_search_documents(processor):
    with patch.object(processor, 'get_embedding') as mock_embedding, \
         patch.object(processor.search_client, 'search') as mock_search:
        
        mock_embedding.return_value = [0.1, 0.2, 0.3]
        mock_search.return_value = [
            {
                "@search.score": 0.95,
                "id": "test_doc_1",
                "content": "test content",
                "file_name": "test.pdf",
                "page_number": 1,
                "image_descriptions": ["chart"],
                "image_ocr_text": ["data"]
            }
        ]
        
        results = await processor.search_documents("test query")
        assert len(results) == 1
        assert results[0]["score"] == 0.95