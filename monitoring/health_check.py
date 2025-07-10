import asyncio
import aiohttp
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def check_service_health():
    """서비스 상태 체크"""
    endpoints = [
        "http://localhost:8000/health",
        # 추가 엔드포인트들...
    ]
    
    results = {}
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                async with session.get(endpoint, timeout=10) as response:
                    if response.status == 200:
                        results[endpoint] = "healthy"
                    else:
                        results[endpoint] = f"unhealthy (status: {response.status})"
            except Exception as e:
                results[endpoint] = f"error: {str(e)}"
    
    return results