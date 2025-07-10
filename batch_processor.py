import asyncio
import logging
from datetime import datetime
from config import AzureConfig
from azure_pdf_processor import AzurePDFProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pdf_processing_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def batch_process():
    """배치 처리 메인 함수"""
    try:
        logger.info("배치 처리 시작")
        
        # 설정 로드
        config = AzureConfig.get_config_dict()
        processor = AzurePDFProcessor(config)
        
        # 인덱스 생성
        logger.info("AI Search 인덱스 확인/생성")
        await processor.create_search_index()
        
        # PDF 처리
        logger.info("PDF 파일 처리 시작")
        await processor.process_all_pdfs_in_container("guide-data")
        
        logger.info("배치 처리 완료")
        
    except Exception as e:
        logger.error(f"배치 처리 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(batch_process())