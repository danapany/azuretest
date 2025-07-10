import os
from dotenv import load_dotenv

load_dotenv()

class AzureConfig:
    """Azure 서비스 설정"""
    
    # Storage Account 설정
    STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "your_storage_account")
    STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "your_storage_key")
    
    # Document Intelligence 설정
    DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "https://iap-doc-intelligence.cognitiveservices.azure.com/")
    DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY", "your_doc_intelligence_key")
    
    # Computer Vision 설정 (iap-aiservices-01)
    COMPUTER_VISION_ENDPOINT = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT", "https://iap-aiservices-01.cognitiveservices.azure.com/")
    COMPUTER_VISION_KEY = os.getenv("AZURE_COMPUTER_VISION_KEY", "your_computer_vision_key")
    
    # AI Search 설정
    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://your-search-service.search.windows.net")
    SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "your_search_key")
    SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "pdf-documents-index")
    
    # OpenAI 설정 (임베딩용)
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-openai.openai.azure.com/")
    OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "your_openai_key")
    
    # PDF 컨테이너 설정
    PDF_CONTAINER_NAME = os.getenv("PDF_CONTAINER_NAME", "guide-data")
    
    @classmethod
    def get_config_dict(cls):
        """설정을 딕셔너리로 반환"""
        return {
            "storage_account_name": cls.STORAGE_ACCOUNT_NAME,
            "storage_account_key": cls.STORAGE_ACCOUNT_KEY,
            "doc_intelligence_endpoint": cls.DOC_INTELLIGENCE_ENDPOINT,
            "doc_intelligence_key": cls.DOC_INTELLIGENCE_KEY,
            "computer_vision_endpoint": cls.COMPUTER_VISION_ENDPOINT,
            "computer_vision_key": cls.COMPUTER_VISION_KEY,
            "search_endpoint": cls.SEARCH_ENDPOINT,
            "search_key": cls.SEARCH_KEY,
            "search_index_name": cls.SEARCH_INDEX_NAME,
            "openai_endpoint": cls.OPENAI_ENDPOINT,
            "openai_key": cls.OPENAI_KEY
        }