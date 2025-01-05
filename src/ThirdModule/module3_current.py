# module3.py
import requests
from typing import Optional
import logging
from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# Hugging Face API 정보
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not API_KEY:
    raise ValueError("API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

class AnswerVerifier:
    def verify_answer(self, question: str, choices: dict) -> Optional[str]:
        """주어진 문제와 보기를 바탕으로 정답을 검증"""
        try:
            prompt = self._create_prompt(question, choices)
            headers = {"Authorization": f"Bearer {API_KEY}"}
            
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt}
            )
            response.raise_for_status()
            
            response_data = response.json()
            logger.debug(f"Raw API response: {response_data}")
            
            # API 응답 처리
            generated_text = ""
            if isinstance(response_data, list):
                if response_data and isinstance(response_data[0], dict):
                    generated_text = response_data[0].get('generated_text', '')
                else:
                    generated_text = response_data[0] if response_data else ''
            elif isinstance(response_data, dict):
                generated_text = response_data.get('generated_text', '')
            else:
                generated_text = str(response_data)
            
            verified_answer = self._extract_answer(generated_text)
            logger.info(f"Verified answer: {verified_answer}")
            return verified_answer

        except Exception as e:
            logger.error(f"Error in verify_answer: {e}")
            return None

    def _create_prompt(self, question: str, choices: dict) -> str:
        """검증을 위한 프롬프트 생성"""
        return f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an expert mathematics teacher checking student answers. 
        Please analyze the following question and select the single best answer.
        Output ONLY the letter of the correct answer (A, B, C, or D) without any explanation.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Question: {question}

        A) {choices['A']}
        B) {choices['B']}
        C) {choices['C']}
        D) {choices['D']}

        Select the correct answer letter (A, B, C, or D):
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """.strip()

    def _extract_answer(self, response: str) -> Optional[str]:
        """응답에서 A, B, C, D 중 하나를 추출"""
        response = response.strip().upper()
        valid_answers = {'A', 'B', 'C', 'D'}
        
        # 응답에서 유효한 답안 찾기
        for answer in valid_answers:
            if answer in response:
                return answer
        
        return None
