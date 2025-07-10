from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from config import AzureConfig
from azure_pdf_processor import AzurePDFProcessor

app = FastAPI(title="Azure PDF Search API", version="1.0.0")

# 글로벌 프로세서 인스턴스
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    config = AzureConfig.get_config_dict()
    processor = AzurePDFProcessor(config)
    
    # 인덱스 생성 (존재하지 않는 경우)
    await processor.create_search_index()

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResponse(BaseModel):
    results: List[dict]
    total_results: int
    query: str

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """문서 검색 API"""
    try:
        results = await processor.search_documents(request.query, request.top_k)
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-pdfs")
async def process_pdfs():
    """PDF 처리 API"""
    try:
        await processor.process_all_pdfs_in_container("guide-data")
        return {"message": "PDF 처리 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


# 실행 방법:
# uvicorn web_api:app --reload --host 0.0.0.0 --port 8000