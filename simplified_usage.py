import asyncio
from config import AzureConfig
from azure_pdf_processor import AzurePDFProcessor

async def simple_usage():
    """간단한 사용 예제"""
    
    # 설정 로드
    config = AzureConfig.get_config_dict()
    
    # 프로세서 초기화
    processor = AzurePDFProcessor(config)
    
    print("1. AI Search 인덱스 생성...")
    await processor.create_search_index()
    
    print("2. guide-data 컨테이너의 모든 PDF 처리...")
    await processor.process_all_pdfs_in_container("guide-data")
    
    print("3. 검색 테스트...")
    
    # 다양한 검색 쿼리 테스트
    test_queries = [
        "차트 분석",
        "그래프 데이터",
        "이미지 설명",
        "표 정보",
        "도표 해석"
    ]
    
    for query in test_queries:
        print(f"\n--- 검색: '{query}' ---")
        results = await processor.search_documents(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"결과 {i}: {result['file_name']} (페이지 {result['page_number']})")
            print(f"점수: {result['score']:.4f}")
            print(f"내용: {result['content'][:150]}...")
            if result['image_descriptions']:
                print(f"이미지: {', '.join(result['image_descriptions'])}")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(simple_usage())