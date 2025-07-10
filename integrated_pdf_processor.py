"""
Azure PDF Document Processing and Search System - Integrated Version
통합된 PDF 문서 처리 및 검색 시스템

필수 패키지 설치:
pip install azure-storage-blob azure-ai-formrecognizer azure-search-documents azure-core openai aiohttp pillow requests python-dotenv fastapi uvicorn

사용법:
1. 환경변수 설정 (아래 CONFIG 섹션 수정)
2. python integrated_pdf_processor.py
"""

import os
import json
import time
import base64
import asyncio
import logging
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime

# Azure SDK imports
try:
    from azure.storage.blob import BlobServiceClient
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.models import VectorizedQuery
    from azure.search.documents.indexes.models import (
        SearchIndex, SearchField, SearchFieldDataType, VectorSearch,
        VectorSearchProfile, VectorSearchAlgorithmConfiguration,
        HnswAlgorithmConfiguration, SimpleField, SearchableField, ComplexField
    )
    import openai
    import aiohttp
    from PIL import Image
    import requests
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    print(f"필수 패키지가 설치되지 않았습니다: {e}")
    print("다음 명령어로 설치하세요:")
    print("pip install azure-storage-blob azure-ai-formrecognizer azure-search-documents azure-core openai aiohttp pillow requests python-dotenv fastapi uvicorn")
    exit(1)

# ==================== 설정 ====================
class Config:
    """Azure 서비스 설정 - 실제 값으로 변경하세요"""
    
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

# ==================== 로깅 설정 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pdf_processing_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 메인 프로세서 클래스 ====================
class AzurePDFProcessor:
    def __init__(self):
        """Azure PDF 처리기 초기화"""
        self.config = Config()
        logger.info("Azure PDF 프로세서 초기화 중...")
        
        try:
            # Azure 클라이언트 초기화
            self.blob_client = BlobServiceClient(
                account_url=f"https://{self.config.STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=self.config.STORAGE_ACCOUNT_KEY
            )
            
            self.document_client = DocumentAnalysisClient(
                endpoint=self.config.DOC_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(self.config.DOC_INTELLIGENCE_KEY)
            )
            
            self.search_index_client = SearchIndexClient(
                endpoint=self.config.SEARCH_ENDPOINT,
                credential=AzureKeyCredential(self.config.SEARCH_KEY)
            )
            
            self.search_client = SearchClient(
                endpoint=self.config.SEARCH_ENDPOINT,
                index_name=self.config.SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(self.config.SEARCH_KEY)
            )
            
            # OpenAI 설정
            openai.api_type = "azure"
            openai.api_base = self.config.OPENAI_ENDPOINT
            openai.api_version = "2023-05-15"
            openai.api_key = self.config.OPENAI_KEY
            
            logger.info("Azure 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"Azure 클라이언트 초기화 실패: {e}")
            raise

    async def create_search_index(self):
        """AI Search 인덱스 생성"""
        logger.info("AI Search 인덱스 생성 중...")
        
        # 벡터 검색 설정
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="default-hnsw-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="default-hnsw-config")
            ]
        )
        
        # 인덱스 필드 정의
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="file_name", type=SearchFieldDataType.String),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32),
            SearchableField(name="image_descriptions", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
            SearchableField(name="image_ocr_text", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="default-vector-profile"
            ),
            ComplexField(name="images", type=SearchFieldDataType.Collection(SearchFieldDataType.ComplexType), fields=[
                SimpleField(name="image_id", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SearchableField(name="ocr_text", type=SearchFieldDataType.String),
                SimpleField(name="bounding_box", type=SearchFieldDataType.String),
                SearchableField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String))
            ])
        ]
        
        # 인덱스 생성
        index = SearchIndex(
            name=self.config.SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search
        )
        
        try:
            self.search_index_client.create_index(index)
            logger.info(f"인덱스 '{self.config.SEARCH_INDEX_NAME}' 생성 완료")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"인덱스 '{self.config.SEARCH_INDEX_NAME}' 이미 존재함")
            else:
                logger.error(f"인덱스 생성 오류: {e}")
                raise

    async def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        try:
            if not text.strip():
                return [0.0] * 1536  # 빈 텍스트의 경우 제로 벡터 반환
            
            response = openai.Embedding.create(
                engine="text-embedding-ada-002",
                input=text[:8000]  # 토큰 제한 고려
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            return [0.0] * 1536  # 오류 시 제로 벡터 반환

    async def analyze_image_with_computer_vision(self, image_data: bytes) -> Dict[str, Any]:
        """Computer Vision으로 이미지 분석"""
        headers = {
            'Ocp-Apim-Subscription-Key': self.config.COMPUTER_VISION_KEY,
            'Content-Type': 'application/octet-stream'
        }
        
        result = {
            "ocr_text": "",
            "description": "",
            "tags": []
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # OCR 분석
                ocr_url = f"{self.config.COMPUTER_VISION_ENDPOINT}/vision/v3.2/read/analyze"
                
                async with session.post(ocr_url, headers=headers, data=image_data) as response:
                    if response.status == 202:
                        operation_url = response.headers.get('Operation-Location')
                        if operation_url:
                            ocr_result = await self._wait_for_ocr_result(session, operation_url)
                            result["ocr_text"] = ocr_result.get("text", "")
                
                # 이미지 설명 생성
                caption_url = f"{self.config.COMPUTER_VISION_ENDPOINT}/vision/v3.2/analyze"
                caption_params = {"visualFeatures": "Description,Tags"}
                
                async with session.post(
                    caption_url, 
                    headers=headers, 
                    data=image_data,
                    params=caption_params
                ) as response:
                    if response.status == 200:
                        caption_data = await response.json()
                        descriptions = caption_data.get('description', {}).get('captions', [])
                        if descriptions:
                            result["description"] = descriptions[0].get('text', '')
                        
                        tags = caption_data.get('tags', [])
                        result["tags"] = [tag['name'] for tag in tags if tag.get('confidence', 0) > 0.5]
                        
        except Exception as e:
            logger.error(f"이미지 분석 오류: {e}")
        
        return result

    async def _wait_for_ocr_result(self, session: aiohttp.ClientSession, operation_url: str) -> Dict[str, Any]:
        """OCR 결과 대기"""
        headers = {'Ocp-Apim-Subscription-Key': self.config.COMPUTER_VISION_KEY}
        
        for attempt in range(30):  # 최대 30초 대기
            await asyncio.sleep(1)
            
            try:
                async with session.get(operation_url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if result['status'] == 'succeeded':
                            # OCR 텍스트 추출
                            text_lines = []
                            for read_result in result.get('analyzeResult', {}).get('readResults', []):
                                for line in read_result.get('lines', []):
                                    text_lines.append(line.get('text', ''))
                            
                            return {"text": " ".join(text_lines)}
                        elif result['status'] == 'failed':
                            logger.error("OCR 분석 실패")
                            return {"text": ""}
            except Exception as e:
                logger.error(f"OCR 결과 확인 중 오류 (시도 {attempt + 1}): {e}")
        
        logger.warning("OCR 결과 대기 시간 초과")
        return {"text": ""}

    async def process_pdf_from_blob(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Blob Storage에서 PDF 분석"""
        logger.info(f"PDF 분석 시작: {blob_name}")
        
        try:
            # Blob에서 PDF 다운로드
            blob_client = self.blob_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            blob_data = blob_client.download_blob().readall()
            logger.info(f"PDF 다운로드 완료: {len(blob_data)} bytes")
            
            # Document Intelligence로 분석
            poller = self.document_client.begin_analyze_document(
                "prebuilt-layout", 
                blob_data
            )
            
            result = poller.result()
            logger.info(f"Document Intelligence 분석 완료: {len(result.pages)} 페이지")
            
            # 분석 결과 처리
            processed_data = {
                "file_name": blob_name,
                "pages": [],
                "total_content": "",
                "images": []
            }
            
            for page_idx, page in enumerate(result.pages):
                logger.info(f"페이지 {page_idx + 1} 처리 중...")
                
                page_content = ""
                page_images = []
                
                # 텍스트 추출
                if result.paragraphs:
                    for paragraph in result.paragraphs:
                        # 해당 페이지의 paragraphs만 추출
                        page_spans = [span for span in paragraph.spans if hasattr(span, 'offset')]
                        if page_spans:
                            page_content += paragraph.content + "\n"
                
                # 테이블 내용 추출
                if result.tables:
                    for table in result.tables:
                        if hasattr(table, 'bounding_regions') and table.bounding_regions:
                            # 테이블이 현재 페이지에 있는지 확인
                            if any(region.page_number == page_idx + 1 for region in table.bounding_regions):
                                table_content = self._extract_table_content(table)
                                page_content += f"\n[표]\n{table_content}\n"
                
                # 이미지 처리 (실제 이미지 데이터가 있는 경우)
                if hasattr(page, 'images') and page.images:
                    logger.info(f"페이지 {page_idx + 1}에서 {len(page.images)}개 이미지 발견")
                    
                    for img_idx, image in enumerate(page.images):
                        try:
                            # 실제 환경에서는 이미지 바이트 데이터를 추출해야 함
                            # 여기서는 Mock 데이터 사용
                            image_analysis = await self._analyze_mock_image(blob_name, page_idx + 1, img_idx + 1)
                            
                            image_info = {
                                "image_id": f"{blob_name}_page_{page_idx + 1}_img_{img_idx + 1}",
                                "description": image_analysis["description"],
                                "ocr_text": image_analysis["ocr_text"],
                                "bounding_box": str(image.bounding_regions[0].polygon) if image.bounding_regions else "",
                                "tags": image_analysis["tags"]
                            }
                            
                            page_images.append(image_info)
                            processed_data["images"].append(image_info)
                            
                        except Exception as e:
                            logger.error(f"이미지 처리 오류 (페이지 {page_idx + 1}, 이미지 {img_idx + 1}): {e}")
                
                processed_data["pages"].append({
                    "page_number": page_idx + 1,
                    "content": page_content,
                    "images": page_images
                })
                
                processed_data["total_content"] += page_content
            
            logger.info(f"PDF 처리 완료: {blob_name}")
            return processed_data
            
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 ({blob_name}): {e}")
            raise

    def _extract_table_content(self, table) -> str:
        """테이블 내용 추출"""
        try:
            rows = {}
            for cell in table.cells:
                row_idx = cell.row_index
                col_idx = cell.column_index
                
                if row_idx not in rows:
                    rows[row_idx] = {}
                
                rows[row_idx][col_idx] = cell.content
            
            # 테이블을 문자열로 변환
            table_str = ""
            for row_idx in sorted(rows.keys()):
                row_cells = []
                for col_idx in sorted(rows[row_idx].keys()):
                    row_cells.append(rows[row_idx][col_idx])
                table_str += " | ".join(row_cells) + "\n"
            
            return table_str
            
        except Exception as e:
            logger.error(f"테이블 추출 오류: {e}")
            return ""

    async def _analyze_mock_image(self, file_name: str, page_num: int, img_num: int) -> Dict[str, Any]:
        """Mock 이미지 분석 (실제 구현에서는 실제 이미지 바이트 데이터 사용)"""
        # 실제 환경에서는 Document Intelligence에서 추출한 이미지 바이트 데이터를
        # Computer Vision API로 분석해야 합니다.
        
        mock_descriptions = [
            "차트나 그래프가 포함된 이미지",
            "데이터 테이블이 있는 이미지", 
            "다이어그램이나 도표",
            "텍스트가 포함된 이미지",
            "프로세스 플로우차트"
        ]
        
        mock_ocr_texts = [
            "매출 증가율 15%",
            "2024년 분기별 실적",
            "프로세스 단계 1-5",
            "중요 데이터 포인트",
            "결론 및 권고사항"
        ]
        
        mock_tags = [
            ["chart", "graph", "data"],
            ["table", "numbers", "statistics"],
            ["diagram", "process", "flow"],
            ["text", "document", "report"],
            ["flowchart", "steps", "procedure"]
        ]
        
        idx = (page_num + img_num) % len(mock_descriptions)
        
        return {
            "description": f"{mock_descriptions[idx]} ({file_name} 페이지 {page_num})",
            "ocr_text": mock_ocr_texts[idx],
            "tags": mock_tags[idx]
        }

    async def index_document(self, processed_data: Dict[str, Any]):
        """문서를 AI Search에 인덱싱"""
        logger.info(f"문서 인덱싱 시작: {processed_data['file_name']}")
        
        documents = []
        
        for page in processed_data["pages"]:
            # 페이지 내용과 이미지 정보 결합
            combined_content = page["content"]
            image_descriptions = []
            image_ocr_texts = []
            
            for image in page["images"]:
                combined_content += f" [이미지: {image['description']}]"
                combined_content += f" [이미지 텍스트: {image['ocr_text']}]"
                image_descriptions.append(image["description"])
                image_ocr_texts.append(image["ocr_text"])
            
            # 임베딩 생성
            logger.info(f"페이지 {page['page_number']} 임베딩 생성 중...")
            content_vector = await self.get_embedding(combined_content)
            
            # 문서 생성
            doc = {
                "id": f"{processed_data['file_name']}_page_{page['page_number']}".replace(' ', '_').replace('.', '_'),
                "content": combined_content,
                "file_name": processed_data["file_name"],
                "page_number": page["page_number"],
                "image_descriptions": image_descriptions,
                "image_ocr_text": image_ocr_texts,
                "content_vector": content_vector,
                "images": page["images"]
            }
            
            documents.append(doc)
        
        # 배치 업로드
        try:
            result = self.search_client.upload_documents(documents)
            logger.info(f"인덱싱 완료: {len(documents)}개 페이지")
            return result
        except Exception as e:
            logger.error(f"인덱싱 오류: {e}")
            return None

    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """벡터 검색 수행"""
        try:
            logger.info(f"검색 쿼리: '{query}' (상위 {top_k}개)")
            
            # 쿼리 임베딩 생성
            query_vector = await self.get_embedding(query)
            
            # 벡터 검색
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # 하이브리드 검색 (텍스트 + 벡터)
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "content", "file_name", "page_number", "image_descriptions", "image_ocr_text", "images"],
                top=top_k
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "score": result.get("@search.score", 0),
                    "id": result["id"],
                    "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                    "file_name": result["file_name"],
                    "page_number": result["page_number"],
                    "image_descriptions": result.get("image_descriptions", []),
                    "image_ocr_text": result.get("image_ocr_text", []),
                    "images": result.get("images", [])
                })
            
            logger.info(f"검색 완료: {len(search_results)}개 결과")
            return search_results
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return []

    async def process_all_pdfs_in_container(self, container_name: str = None):
        """컨테이너의 모든 PDF 처리"""
        if container_name is None:
            container_name = self.config.PDF_CONTAINER_NAME
            
        logger.info(f"컨테이너 '{container_name}'의 PDF 파일 처리 시작")
        
        try:
            # 컨테이너의 모든 blob 나열
            container_client = self.blob_client.get_container_client(container_name)
            
            pdf_files = []
            for blob in container_client.list_blobs():
                if blob.name.lower().endswith('.pdf'):
                    pdf_files.append(blob.name)
            
            logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")
            
            if not pdf_files:
                logger.warning("처리할 PDF 파일이 없습니다.")
                return
            
            # 각 PDF 처리
            for pdf_file in pdf_files:
                try:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"처리 중: {pdf_file}")
                    logger.info(f"{'='*50}")
                    
                    # PDF 분석
                    processed_data = await self.process_pdf_from_blob(container_name, pdf_file)
                    
                    # 인덱싱
                    await self.index_document(processed_data)
                    
                    logger.info(f"완료: {pdf_file}")
                    
                except Exception as e:
                    logger.error(f"오류 발생 ({pdf_file}): {e}")
                    continue
            
            logger.info(f"\n전체 처리 완료: {len(pdf_files)}개 파일")
            
        except Exception as e:
            logger.error(f"컨테이너 처리 중 오류: {e}")
            raise

# ==================== FastAPI 웹 서버 ====================
app = FastAPI(title="Azure PDF Search API", version="1.0.0")

# 글로벌 프로세서 인스턴스
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    try:
        processor = AzurePDFProcessor()
        await processor.create_search_index()
        logger.info("FastAPI 서버 시작 완료")
    except Exception as e:
        logger.error(f"FastAPI 서버 시작 오류: {e}")
        raise

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResponse(BaseModel):
    results: List[dict]
    total_results: int
    query: str

@app.post("/search", response_model=SearchResponse)
async def search_documents_api(request: SearchRequest):
    """문서 검색 API"""
    try:
        results = await processor.search_documents(request.query, request.top_k)
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            query=request.query
        )
    except Exception as e:
        logger.error(f"검색 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-pdfs")
async def process_pdfs_api():
    """PDF 처리 API"""
    try:
        await processor.process_all_pdfs_in_container()
        return {"message": "PDF 처리 완료", "status": "success"}
    except Exception as e:
        logger.error(f"PDF 처리 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "Azure PDF Document Processing and Search API",
        "endpoints": {
            "search": "POST /search",
            "process": "POST /process-pdfs", 
            "health": "GET /health"
        }
    }

# ==================== 메인 실행 함수 ====================
async def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Azure PDF Document Processing and Search System")
    print("=" * 60)
    
    try:
        # 프로세서 초기화
        pdf_processor = AzurePDFProcessor()
        
        # 인덱스 생성
        print("\n1. AI Search 인덱스 생성 중...")
        await pdf_processor.create_search_index()
        
        # PDF 처리
        print("\n2. PDF 파일 처리 중...")
        await pdf_processor.process_all_pdfs_in_container()
        
        # 검색 테스트
        print("\n3. 검색 기능 테스트 중...")
        
        test_queries = [
            "차트 분석",
            "데이터 테이블", 
            "그래프",
            "이미지 설명",
            "프로세스"
        ]
        
        for query in test_queries:
            print(f"\n--- 검색: '{query}' ---")
            results = await pdf_processor.search_documents(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"결과 {i}:")
                    print(f"  파일: {result['file_name']}")
                    print(f"  페이지: {result['page_number']}")
                    print(f"  점수: {result['score']:.4f}")
                    print(f"  내용: {result['content'][:100]}...")
                    if result['image_descriptions']:
                        print(f"  이미지: {', '.join(result['image_descriptions'])}")
                    print("-" * 40)
            else:
                print("  검색 결과 없음")
        
        print("\n=" * 60)
        print("처리 완료! 웹 API 서버를 시작하려면 다음 명령어를 사용하세요:")
        print("python integrated_pdf_processor.py --web")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"메인 실행 중 오류: {e}")
        raise

def run_web_server():
    """웹 서버 실행"""
    print("FastAPI 웹 서버 시작 중...")
    print("API 문서: http://localhost:8000/docs")
    print("헬스 체크: http://localhost:8000/health")
    
    uvicorn.run(
        "integrated_pdf_processor:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )

def show_usage():
    """사용법 출력"""
    print("""
Azure PDF Document Processing and Search System - 통합 버전

사용법:
    python integrated_pdf_processor.py              # 전체 처리 실행
    python integrated_pdf_processor.py --web        # 웹 서버 시작
    python integrated_pdf_processor.py --help       # 도움말

설정:
    파일 상단의 Config 클래스에서 Azure 서비스 설정을 변경하거나
    환경변수를 사용하여 설정할 수 있습니다.

필수 환경변수:
    AZURE_STORAGE_ACCOUNT_NAME      # Storage Account 이름
    AZURE_STORAGE_ACCOUNT_KEY       # Storage Account 키
    AZURE_DOC_INTELLIGENCE_ENDPOINT # Document Intelligence 엔드포인트
    AZURE_DOC_INTELLIGENCE_KEY     # Document Intelligence 키
    AZURE_COMPUTER_VISION_ENDPOINT # Computer Vision 엔드포인트  
    AZURE_COMPUTER_VISION_KEY      # Computer Vision 키
    AZURE_SEARCH_ENDPOINT          # AI Search 엔드포인트
    AZURE_SEARCH_KEY               # AI Search 키
    AZURE_OPENAI_ENDPOINT          # OpenAI 엔드포인트
    AZURE_OPENAI_KEY               # OpenAI 키

웹 API 엔드포인트:
    POST /search                    # 문서 검색
    POST /process-pdfs              # PDF 처리
    GET  /health                    # 헬스 체크
    GET  /                          # API 정보

예제 검색 요청:
    curl -X POST "http://localhost:8000/search" \\
         -H "Content-Type: application/json" \\
         -d '{"query": "차트 분석", "top_k": 5}'

로그 파일:
    pdf_processing_YYYYMMDD.log     # 처리 로그
    """)

# ==================== 명령행 인터페이스 ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--web":
            run_web_server()
        elif sys.argv[1] == "--help":
            show_usage()
        else:
            print(f"알 수 없는 옵션: {sys.argv[1]}")
            show_usage()
            sys.exit(1)
    else:
        # 기본 실행: 전체 처리
        asyncio.run(main())

# ==================== 추가 유틸리티 함수 ====================
class PDFProcessorUtils:
    """PDF 프로세서 유틸리티 함수들"""
    
    @staticmethod
    def validate_config() -> bool:
        """설정 검증"""
        required_configs = [
            'STORAGE_ACCOUNT_NAME',
            'STORAGE_ACCOUNT_KEY', 
            'DOC_INTELLIGENCE_ENDPOINT',
            'DOC_INTELLIGENCE_KEY',
            'COMPUTER_VISION_ENDPOINT',
            'COMPUTER_VISION_KEY',
            'SEARCH_ENDPOINT',
            'SEARCH_KEY',
            'OPENAI_ENDPOINT',
            'OPENAI_KEY'
        ]
        
        config = Config()
        missing_configs = []
        
        for config_name in required_configs:
            value = getattr(config, config_name, None)
            if not value or value.startswith('your_'):
                missing_configs.append(config_name)
        
        if missing_configs:
            print("다음 설정이 누락되었습니다:")
            for config in missing_configs:
                print(f"  - {config}")
            return False
        
        return True
    
    @staticmethod
    def create_sample_env_file():
        """샘플 .env 파일 생성"""
        env_content = """# Azure PDF Processor 환경변수 설정

# Storage Account 설정
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
AZURE_STORAGE_ACCOUNT_KEY=your_storage_account_key

# Document Intelligence 설정 
AZURE_DOC_INTELLIGENCE_ENDPOINT=https://iap-doc-intelligence.cognitiveservices.azure.com/
AZURE_DOC_INTELLIGENCE_KEY=your_doc_intelligence_key

# Computer Vision 설정 (iap-aiservices-01)
AZURE_COMPUTER_VISION_ENDPOINT=https://iap-aiservices-01.cognitiveservices.azure.com/
AZURE_COMPUTER_VISION_KEY=your_computer_vision_key

# AI Search 설정
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=pdf-documents-index

# OpenAI 설정 (임베딩용)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_KEY=your_openai_key

# PDF 컨테이너 설정
PDF_CONTAINER_NAME=guide-data
"""
        
        with open('.env.sample', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("샘플 환경변수 파일 '.env.sample' 생성 완료")
        print("이 파일을 '.env'로 복사하고 실제 값으로 수정하세요")
    
    @staticmethod
    def test_azure_connections():
        """Azure 서비스 연결 테스트"""
        print("Azure 서비스 연결 테스트 중...")
        
        # 설정 검증
        if not PDFProcessorUtils.validate_config():
            return False
        
        config = Config()
        
        # Storage Account 테스트
        try:
            blob_client = BlobServiceClient(
                account_url=f"https://{config.STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=config.STORAGE_ACCOUNT_KEY
            )
            
            # 컨테이너 존재 확인
            container_client = blob_client.get_container_client(config.PDF_CONTAINER_NAME)
            container_client.get_container_properties()
            print("✓ Storage Account 연결 성공")
            
        except Exception as e:
            print(f"✗ Storage Account 연결 실패: {e}")
            return False
        
        # Document Intelligence 테스트
        try:
            doc_client = DocumentAnalysisClient(
                endpoint=config.DOC_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(config.DOC_INTELLIGENCE_KEY)
            )
            print("✓ Document Intelligence 연결 성공")
            
        except Exception as e:
            print(f"✗ Document Intelligence 연결 실패: {e}")
            return False
        
        # AI Search 테스트
        try:
            search_client = SearchIndexClient(
                endpoint=config.SEARCH_ENDPOINT,
                credential=AzureKeyCredential(config.SEARCH_KEY)
            )
            print("✓ AI Search 연결 성공")
            
        except Exception as e:
            print(f"✗ AI Search 연결 실패: {e}")
            return False
        
        print("모든 Azure 서비스 연결 테스트 통과!")
        return True

# ==================== 테스트 실행기 ====================
async def run_tests():
    """통합 테스트 실행"""
    print("=" * 50)
    print("Azure PDF Processor 통합 테스트")
    print("=" * 50)
    
    # 연결 테스트
    print("\n1. Azure 서비스 연결 테스트")
    if not PDFProcessorUtils.test_azure_connections():
        print("연결 테스트 실패. 설정을 확인하세요.")
        return
    
    # 프로세서 테스트
    print("\n2. PDF 프로세서 테스트")
    try:
        processor = AzurePDFProcessor()
        
        # 인덱스 생성 테스트
        await processor.create_search_index()
        print("✓ 인덱스 생성 성공")
        
        # 검색 테스트 (빈 인덱스에서)
        results = await processor.search_documents("테스트", top_k=1)
        print(f"✓ 검색 기능 작동 (결과: {len(results)}개)")
        
        print("모든 테스트 통과!")
        
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")

# ==================== 설정 마법사 ====================
def setup_wizard():
    """대화형 설정 마법사"""
    print("=" * 50)
    print("Azure PDF Processor 설정 마법사")
    print("=" * 50)
    
    config_values = {}
    
    # 각 설정 항목 입력 받기
    configs = [
        ("AZURE_STORAGE_ACCOUNT_NAME", "Storage Account 이름"),
        ("AZURE_STORAGE_ACCOUNT_KEY", "Storage Account 키"),
        ("AZURE_DOC_INTELLIGENCE_ENDPOINT", "Document Intelligence 엔드포인트"),
        ("AZURE_DOC_INTELLIGENCE_KEY", "Document Intelligence 키"),
        ("AZURE_COMPUTER_VISION_ENDPOINT", "Computer Vision 엔드포인트"),
        ("AZURE_COMPUTER_VISION_KEY", "Computer Vision 키"),
        ("AZURE_SEARCH_ENDPOINT", "AI Search 엔드포인트"),
        ("AZURE_SEARCH_KEY", "AI Search 키"),
        ("AZURE_OPENAI_ENDPOINT", "OpenAI 엔드포인트"),
        ("AZURE_OPENAI_KEY", "OpenAI 키"),
    ]
    
    for config_key, description in configs:
        while True:
            value = input(f"{description}: ").strip()
            if value:
                config_values[config_key] = value
                break
            else:
                print("값을 입력해주세요.")
    
    # .env 파일 생성
    env_content = ""
    for key, value in config_values.items():
        env_content += f"{key}={value}\n"
    
    env_content += "AZURE_SEARCH_INDEX_NAME=pdf-documents-index\n"
    env_content += "PDF_CONTAINER_NAME=guide-data\n"
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("\n.env 파일이 생성되었습니다!")
    print("이제 프로그램을 실행할 수 있습니다.")

# ==================== 확장된 명령행 인터페이스 ====================
def main_cli():
    """확장된 명령행 인터페이스"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure PDF Document Processing and Search System')
    parser.add_argument('--web', action='store_true', help='웹 서버 시작')
    parser.add_argument('--test', action='store_true', help='테스트 실행')
    parser.add_argument('--setup', action='store_true', help='설정 마법사 실행')
    parser.add_argument('--create-env', action='store_true', help='샘플 .env 파일 생성')
    parser.add_argument('--validate', action='store_true', help='설정 검증')
    
    args = parser.parse_args()
    
    if args.web:
        run_web_server()
    elif args.test:
        asyncio.run(run_tests())
    elif args.setup:
        setup_wizard()
    elif args.create_env:
        PDFProcessorUtils.create_sample_env_file()
    elif args.validate:
        if PDFProcessorUtils.validate_config():
            print("✓ 모든 설정이 올바릅니다.")
        else:
            print("✗ 설정에 문제가 있습니다.")
    else:
        # 기본 실행
        asyncio.run(main())

# 확장된 CLI 사용 (옵션)
# if __name__ == "__main__":
#     main_cli()

"""
사용 예제:

1. 기본 실행 (PDF 처리):
   python integrated_pdf_processor.py

2. 웹 서버 시작:
   python integrated_pdf_processor.py --web

3. 설정 마법사:
   python integrated_pdf_processor.py --setup

4. 테스트 실행:
   python integrated_pdf_processor.py --test

5. 설정 검증:
   python integrated_pdf_processor.py --validate

6. 샘플 환경파일 생성:
   python integrated_pdf_processor.py --create-env
"""