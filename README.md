# Azure PDF Document Processing and Search System

Azure 서비스를 활용한 PDF 문서 처리 및 검색 시스템입니다.

## 기능

- PDF 문서에서 텍스트 및 이미지 추출
- 이미지 OCR 및 설명 생성
- 벡터 기반 의미 검색
- RESTful API 제공

## 필수 Azure 서비스

1. **Azure Document Intelligence** (iap-doc-intelligence)
2. **Azure Computer Vision** (iap-aiservices-01)
3. **Azure AI Search**
4. **Azure OpenAI**
5. **Azure Storage Account** (guide-data 컨테이너)

## 설치

```bash
git clone <repository>
cd azure-pdf-processor
pip install -r requirements.txt
```

## 설정

1. `.env` 파일 생성:
```bash
cp .env.template .env
```

2. Azure 서비스 키 설정:
```env
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
AZURE_STORAGE_ACCOUNT_KEY=your_key
AZURE_DOC_INTELLIGENCE_ENDPOINT=https://iap-doc-intelligence.cognitiveservices.azure.com/
AZURE_DOC_INTELLIGENCE_KEY=your_key
AZURE_COMPUTER_VISION_ENDPOINT=https://iap-aiservices-01.cognitiveservices.azure.com/
AZURE_COMPUTER_VISION_KEY=your_key
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_KEY=your_key
```

## 사용법

### 1. 배치 처리
```bash
python batch_processor.py
```

### 2. 웹 API 서버
```bash
uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
```

### 3. 간단한 사용
```bash
python simplified_usage.py
```

## API 엔드포인트

- `POST /search` - 문서 검색
- `POST /process-pdfs` - PDF 처리
- `GET /health` - 헬스 체크

## Docker 배포

```bash
docker-compose up -d
```

## 테스트

```bash
pytest test/
```

## 모니터링

```bash
python monitoring/health_check.py
```

## 라이선스

MIT License
