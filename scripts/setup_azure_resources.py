"""
Azure 리소스 설정을 위한 스크립트
실제 Azure CLI 또는 Azure SDK를 사용하여 리소스를 생성할 수 있습니다.
"""

import subprocess
import json

def create_azure_resources():
    """Azure 리소스 생성 스크립트"""
    
    # 리소스 그룹 생성
    subprocess.run([
        "az", "group", "create",
        "--name", "rg-pdf-processor",
        "--location", "koreacentral"
    ])
    
    # Document Intelligence 생성
    subprocess.run([
        "az", "cognitiveservices", "account", "create",
        "--name", "iap-doc-intelligence",
        "--resource-group", "rg-pdf-processor",
        "--kind", "FormRecognizer",
        "--sku", "S0",
        "--location", "koreacentral"
    ])
    
    # Computer Vision 생성 (Multi-service account)
    subprocess.run([
        "az", "cognitiveservices", "account", "create",
        "--name", "iap-aiservices-01",
        "--resource-group", "rg-pdf-processor",
        "--kind", "CognitiveServices",
        "--sku", "S0",
        "--location", "koreacentral"
    ])
    
    # AI Search 생성
    subprocess.run([
        "az", "search", "service", "create",
        "--name", "pdf-search-service",
        "--resource-group", "rg-pdf-processor",
        "--sku", "standard",
        "--location", "koreacentral"
    ])
    
    # Storage Account 생성
    subprocess.run([
        "az", "storage", "account", "create",
        "--name", "pdfstorageaccount",
        "--resource-group", "rg-pdf-processor",
        "--location", "koreacentral",
        "--sku", "Standard_LRS"
    ])
    
    # Container 생성
    subprocess.run([
        "az", "storage", "container", "create",
        "--name", "guide-data",
        "--account-name", "pdfstorageaccount"
    ])

if __name__ == "__main__":
    create_azure_resources()