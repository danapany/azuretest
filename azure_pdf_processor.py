import asyncio
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
import base64
import hashlib

# Azure SDK 임포트
try:
    from azure.storage.blob.aio import BlobServiceClient
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.aio import SearchClient
    from azure.search.documents.indexes.aio import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex, 
        SearchField, 
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration
    )
    from azure.ai.formrecognizer.aio import DocumentAnalysisClient
    from azure.ai.formrecognizer import AnalyzeResult
except ImportError as e:
    logging.error(f"Azure SDK 모듈 임포트 실패: {e}")
    logging.error("다음 명령어로 필요한 패키지를 설치하세요:")
    logging.error("pip install azure-storage-blob azure-search-documents azure-ai-formrecognizer aiohttp")
    raise

logger = logging.getLogger(__name__)

class AzurePDFProcessor:
    """Azure를 사용한 PDF 처리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: Azure 설정 딕셔너리
        """
        self.config = config
        self.validate_config()
        
        # Azure 클라이언트 초기화
        self.blob_service_client = None
        self.search_client = None
        self.search_index_client = None
        self.document_analysis_client = None
        
        # 인덱스 이름
        self.index_name = config.get('search_index_name', 'pdf-documents')
        
    def validate_config(self):
        """설정 유효성 검사"""
        required_keys = [
            'storage_connection_string',
            'search_service_name',
            'search_api_key',
            'form_recognizer_endpoint',
            'form_recognizer_key'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"필수 설정 키 '{key}'가 누락되었습니다.")
            
        logger.info("설정 유효성 검사 완료")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.close_clients()
    
    async def initialize_clients(self):
        """Azure 클라이언트들 초기화"""
        try:
            # Blob Storage 클라이언트
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.config['storage_connection_string']
            )
            
            # Azure Search 클라이언트들
            search_endpoint = f"https://{self.config['search_service_name']}.search.windows.net"
            credential = AzureKeyCredential(self.config['search_api_key'])
            
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=self.index_name,
                credential=credential
            )
            
            self.search_index_client = SearchIndexClient(
                endpoint=search_endpoint,
                credential=credential
            )
            
            # Form Recognizer 클라이언트
            self.document_analysis_client = DocumentAnalysisClient(
                endpoint=self.config['form_recognizer_endpoint'],
                credential=AzureKeyCredential(self.config['form_recognizer_key'])
            )
            
            logger.info("Azure 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"Azure 클라이언트 초기화 실패: {e}")
            raise
    
    async def close_clients(self):
        """클라이언트들 정리"""
        clients = [
            self.blob_service_client,
            self.search_client,
            self.search_index_client,
            self.document_analysis_client
        ]
        
        for client in clients:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"클라이언트 정리 중 오류: {e}")
    
    async def create_search_index(self):
        """AI Search 인덱스 생성"""
        try:
            # 인덱스 스키마 정의
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchableField(name="filename", type=SearchFieldDataType.String),
                SimpleField(name="page_number", type=SearchFieldDataType.Int32),
                SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset),
                SimpleField(name="file_size", type=SearchFieldDataType.Int64),
                SearchableField(name="metadata", type=SearchFieldDataType.String),
            ]
            
            # 인덱스 생성
            index = SearchIndex(
                name=self.index_name,
                fields=fields
            )
            
            # 인덱스 존재 확인
            try:
                await self.search_index_client.get_index(self.index_name)
                logger.info(f"인덱스 '{self.index_name}'가 이미 존재합니다.")
            except Exception:
                # 인덱스가 없으면 생성
                await self.search_index_client.create_index(index)
                logger.info(f"인덱스 '{self.index_name}' 생성 완료")
                
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            raise
    
    async def get_blob_list(self, container_name: str) -> List[str]:
        """컨테이너의 PDF 파일 목록 가져오기"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_list = []
            
            async for blob in container_client.list_blobs():
                if blob.name.lower().endswith('.pdf'):
                    blob_list.append(blob.name)
            
            logger.info(f"컨테이너 '{container_name}'에서 {len(blob_list)}개 PDF 파일 발견")
            return blob_list
            
        except Exception as e:
            logger.error(f"블롭 목록 가져오기 실패: {e}")
            raise
    
    async def download_blob(self, container_name: str, blob_name: str) -> bytes:
        """블롭 다운로드"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_data = await blob_client.download_blob()
            content = await blob_data.readall()
            
            logger.info(f"블롭 '{blob_name}' 다운로드 완료 ({len(content)} bytes)")
            return content
            
        except Exception as e:
            logger.error(f"블롭 다운로드 실패 ({blob_name}): {e}")
            raise
    
    async def extract_text_from_pdf(self, pdf_content: bytes, filename: str) -> List[Dict]:
        """PDF에서 텍스트 추출"""
        try:
            # Form Recognizer를 사용한 텍스트 추출
            poller = await self.document_analysis_client.begin_analyze_document(
                "prebuilt-layout",
                pdf_content
            )
            
            result = await poller.result()
            
            # 페이지별 텍스트 추출
            pages_content = []
            
            for page_num, page in enumerate(result.pages, 1):
                page_text = ""
                
                # 텍스트 라인 추출
                if hasattr(result, 'paragraphs'):
                    for paragraph in result.paragraphs:
                        if any(page_num - 1 in span.page_number for span in paragraph.spans):
                            page_text += paragraph.content + "\n"
                
                # 백업: 라인별 텍스트 추출
                if not page_text.strip():
                    for line in page.lines:
                        page_text += line.content + "\n"
                
                if page_text.strip():
                    pages_content.append({
                        'page_number': page_num,
                        'content': page_text.strip(),
                        'filename': filename
                    })
            
            logger.info(f"PDF '{filename}'에서 {len(pages_content)}개 페이지 텍스트 추출 완료")
            return pages_content
            
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 실패 ({filename}): {e}")
            raise
    
    async def index_document(self, doc_content: Dict):
        """문서를 AI Search에 인덱싱"""
        try:
            # 문서 ID 생성
            doc_id = hashlib.md5(
                f"{doc_content['filename']}_page_{doc_content['page_number']}".encode()
            ).hexdigest()
            
            # 인덱스에 추가할 문서 데이터
            search_doc = {
                "id": doc_id,
                "content": doc_content['content'],
                "filename": doc_content['filename'],
                "page_number": doc_content['page_number'],
                "created_date": datetime.now().isoformat(),
                "file_size": len(doc_content['content']),
                "metadata": json.dumps({
                    "source": "pdf_processor",
                    "processed_date": datetime.now().isoformat()
                })
            }
            
            # 문서 업로드
            await self.search_client.upload_documents([search_doc])
            
            logger.debug(f"문서 인덱싱 완료: {doc_content['filename']} (페이지 {doc_content['page_number']})")
            
        except Exception as e:
            logger.error(f"문서 인덱싱 실패: {e}")
            raise
    
    async def process_single_pdf(self, container_name: str, blob_name: str):
        """단일 PDF 파일 처리"""
        try:
            logger.info(f"PDF 처리 시작: {blob_name}")
            
            # PDF 다운로드
            pdf_content = await self.download_blob(container_name, blob_name)
            
            # 텍스트 추출
            pages_content = await self.extract_text_from_pdf(pdf_content, blob_name)
            
            # 각 페이지 인덱싱
            for page_content in pages_content:
                await self.index_document(page_content)
            
            logger.info(f"PDF 처리 완료: {blob_name} ({len(pages_content)} 페이지)")
            
        except Exception as e:
            logger.error(f"PDF 처리 실패 ({blob_name}): {e}")
            raise
    
    async def process_all_pdfs_in_container(self, container_name: str):
        """컨테이너의 모든 PDF 파일 처리"""
        try:
            # 클라이언트 초기화
            if not self.blob_service_client:
                await self.initialize_clients()
            
            # PDF 파일 목록 가져오기
            blob_list = await self.get_blob_list(container_name)
            
            if not blob_list:
                logger.warning(f"컨테이너 '{container_name}'에 PDF 파일이 없습니다.")
                return
            
            # 각 PDF 파일 처리
            for blob_name in blob_list:
                try:
                    await self.process_single_pdf(container_name, blob_name)
                except Exception as e:
                    logger.error(f"PDF 파일 처리 중 오류 발생 ({blob_name}): {e}")
                    continue
            
            logger.info(f"모든 PDF 처리 완료: {len(blob_list)}개 파일")
            
        except Exception as e:
            logger.error(f"배치 PDF 처리 실패: {e}")
            raise
        finally:
            # 클라이언트 정리
            await self.close_clients()
    
    async def search_documents(self, query: str, top: int = 10) -> List[Dict]:
        """문서 검색"""
        try:
            if not self.search_client:
                await self.initialize_clients()
            
            results = await self.search_client.search(
                search_text=query,
                top=top,
                select=["id", "content", "filename", "page_number", "created_date"]
            )
            
            search_results = []
            async for result in results:
                search_results.append({
                    "id": result["id"],
                    "content": result["content"],
                    "filename": result["filename"],
                    "page_number": result["page_number"],
                    "created_date": result["created_date"],
                    "score": result.get("@search.score", 0)
                })
            
            logger.info(f"검색 완료: '{query}' -> {len(search_results)}개 결과")
            return search_results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            raise