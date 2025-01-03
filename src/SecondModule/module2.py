import pandas as pd
import requests
from typing import Tuple, Optional
from dataclasses import dataclass
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

base_path = os.path.dirname(os.path.abspath(__file__))
misconception_csv_path = os.path.join(base_path, 'misconception_mapping.csv')

if not API_KEY:
    raise ValueError("API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

#유사 문제 생성기 클래스

@dataclass
class GeneratedQuestion:
    question: str
    choices: dict
    correct_answer: str
    explanation: str

class SimilarQuestionGenerator:
    def __init__(self, misconception_csv_path: str = 'misconception_mapping.csv'):
        """
        Initialize the generator by loading the misconception mapping and the language model.
        """
        self._load_data(misconception_csv_path)

    def _load_data(self, misconception_csv_path: str):
        logger.info("Loading misconception mapping...")
        self.misconception_df = pd.read_csv(misconception_csv_path)

    def get_misconception_text(self, misconception_id: float) -> Optional[str]:
        # MisconceptionId를 받아 해당 ID에 매칭되는 오개념 설명 텍스트를 반환합니다
        """Retrieve the misconception text based on the misconception ID."""
        if pd.isna(misconception_id):  # NaN 체크
            logger.warning("Received NaN for misconception_id.")
            return "No misconception provided."
        
        try:
            row = self.misconception_df[self.misconception_df['MisconceptionId'] == int(misconception_id)]
            if not row.empty:
                return row.iloc[0]['MisconceptionName']
        except ValueError as e:
            logger.error(f"Error processing misconception_id: {e}")
        
        logger.warning(f"No misconception found for ID: {misconception_id}")
        return "Misconception not found."

    def generate_prompt(self, construct_name: str, subject_name: str, question_text: str, correct_answer_text: str, wrong_answer_text: str, misconception_text: str) -> str:
        """Create a prompt for the language model."""
        #문제 생성을 위한 프롬프트 텍스트를 생성
        logger.info("Generating prompt...")
        misconception_clause = (f"that targets the following misconception: \"{misconception_text}\"." if misconception_text != "There is no misconception" else "")
        prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are an educational assistant designed to generate multiple-choice questions {misconception_clause}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            You need to create a similar multiple-choice question based on the following details:

            Construct Name: {construct_name}
            Subject Name: {subject_name}
            Question Text: {question_text}
            Correct Answer: {correct_answer_text}
            Wrong Answer: {wrong_answer_text}

            Please follow this output format:
            ---
            Question: <Your Question Text>
            A) <Choice A>
            B) <Choice B>
            C) <Choice C>
            D) <Choice D>
            Correct Answer: <Correct Choice (e.g., A)>
            Explanation: <Brief explanation for the correct answer>
            ---
            Ensure that the question is conceptually similar but not identical to the original. Ensure clarity and educational value.
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """.strip()
        logger.debug(f"Generated prompt: {prompt}")
        return prompt

    # def call_model_api(self, prompt: str) -> str:
    #     """Hugging Face API 호출"""
    #     logger.info("Calling Hugging Face API...")
    #     headers = {"Authorization": f"Bearer {API_KEY}"}
        
    #     try:
    #         response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    #         response.raise_for_status()  # HTTP 에러 체크
            
    #         # 응답 내용 로깅
    #         logger.debug(f"API Response status: {response.status_code}")
    #         logger.debug(f"API Response headers: {response.headers}")
    #         logger.debug(f"API Response content: {response.text}")
            
    #         # JSON 파싱
    #         try:
    #             response_data = response.json()
    #             logger.info("Successfully parsed JSON response")
                
    #             # API 응답 구조 확인
    #             if isinstance(response_data, list) and len(response_data) > 0:
    #                 generated_text = response_data[0].get('generated_text', '')
    #             else:
    #                 generated_text = response_data.get('generated_text', '')
                    
    #             if not generated_text:
    #                 logger.warning("No generated text in response")
    #                 return "Error: No text generated"
                    
    #             return generated_text
                
    #         except ValueError as e:
    #             logger.error(f"JSON parsing error: {e}")
    #             return f"Error: Invalid JSON response - {str(e)}"
                
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"API request failed: {e}")
    #         return f"Error: API request failed - {str(e)}"
    
    def call_model_api(self, prompt: str) -> str:
        """Hugging Face API 호출"""
        logger.info("Calling Hugging Face API...")
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            response.raise_for_status()
            
            response_data = response.json()
            logger.debug(f"Raw API response: {response_data}")
            
            # API 응답이 리스트인 경우 처리
            if isinstance(response_data, list):
                if response_data and isinstance(response_data[0], dict):
                    generated_text = response_data[0].get('generated_text', '')
                else:
                    generated_text = response_data[0] if response_data else ''
            # API 응답이 딕셔너리인 경우 처리
            elif isinstance(response_data, dict):
                generated_text = response_data.get('generated_text', '')
            else:
                generated_text = str(response_data)
                
            logger.info(f"Generated text: {generated_text}")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in call_model_api: {e}")
            raise
    def parse_model_output(self, output: str) -> GeneratedQuestion:
        if not isinstance(output, str):
            logger.error(f"Invalid output format: {type(output)}. Expected string.")
            raise ValueError("Model output is not a string.")

        logger.info(f"Parsing output: {output}")
        output_lines = output.strip().splitlines()
        logger.debug(f"Split output into lines: {output_lines}")

        question, choices, correct_answer, explanation = "", {}, "", ""

        for line in output_lines:
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
            elif line.startswith("A)"):
                choices["A"] = line[2:].strip()
            elif line.startswith("B)"):
                choices["B"] = line[2:].strip()
            elif line.startswith("C)"):
                choices["C"] = line[2:].strip()
            elif line.startswith("D)"):
                choices["D"] = line[2:].strip()
            elif line.lower().startswith("correct answer:"):
                correct_answer = line.split(":", 1)[1].strip()
            elif line.lower().startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()

        if not question or len(choices) < 4 or not correct_answer or not explanation:
            logger.warning("Incomplete generated question.")
        return GeneratedQuestion(question, choices, correct_answer, explanation)

    # def generate_similar_question_with_text(self, construct_name: str, subject_name: str, question_text: str, correct_answer_text: str, wrong_answer_text: str, misconception_id: float) -> Tuple[Optional[GeneratedQuestion], Optional[str]]:
    #     # 유사 문제 생성 
    #     logger.info("Calling generate_similar_question_with_text...")
    #     misconception_text = self.get_misconception_text(misconception_id)
    #     if not misconception_text:
    #         logger.info("Skipping question generation due to lack of misconception.")
    #         return None, None

    #     prompt = self.generate_prompt(construct_name, subject_name, question_text, correct_answer_text, wrong_answer_text, misconception_text)
    #     print(prompt)
    #     logger.info(f"Generated prompt: {prompt}")

    #     generated_text = None  # 기본값으로 초기화
    #     try:
    #         # API 호출
    #         generated_text = self.call_model_api(prompt)
    #         logger.info(f"Generated text from API: {generated_text}")

    #         # 파싱
    #         generated_question = self.parse_model_output(generated_text)
    #         logger.info(f"Generated question object: {generated_question}")
    #         return generated_question, generated_text

    #     except Exception as e:
    #         logger.error(f"Failed to generate question: {e}")
    #         logger.debug(f"API output for debugging: {generated_text}")
    #         return None, generated_text

    def generate_similar_question_with_text(self, construct_name: str, subject_name: str, question_text: str, correct_answer_text: str, wrong_answer_text: str, misconception_id: float) -> Tuple[Optional[GeneratedQuestion], Optional[str]]:
        logger.info("generate_similar_question_with_text initiated")

        # 예외 처리 추가
        try:
            misconception_text = self.get_misconception_text(misconception_id)
            logger.info(f"Misconception text retrieved: {misconception_text}")
        except Exception as e:
            logger.error(f"Error retrieving misconception text: {e}")
            return None, None

        if not misconception_text:
            logger.info("Skipping question generation due to lack of misconception.")
            return None, None

        prompt = self.generate_prompt(construct_name, subject_name, question_text, correct_answer_text, wrong_answer_text, misconception_text)
        logger.info(f"Generated prompt: {prompt}")

        generated_text = None  # 기본값으로 초기화
        try:
            logger.info("Calling call_model_api...")
            generated_text = self.call_model_api(prompt)
            logger.info(f"Generated text from API: {generated_text}")

            # 파싱
            generated_question = self.parse_model_output(generated_text)
            logger.info(f"Generated question object: {generated_question}")
            return generated_question, generated_text

        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            logger.debug(f"API output for debugging: {generated_text}")
            return None, generated_text

