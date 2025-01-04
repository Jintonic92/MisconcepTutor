import requests
import json
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# Hugging Face API 정보
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# 프롬프트 설정
prompt = "Explain the concept of gravitational force."

# API 요청
headers = {"Authorization": f"Bearer {API_KEY}"}
data = {"inputs": prompt}

response = requests.post(API_URL, headers=headers, json=data)

# 결과 출력
if response.status_code == 200:
    result = response.json()
    print("Response:", result)
else:
    print(f"Error: {response.status_code}, {response.text}")
