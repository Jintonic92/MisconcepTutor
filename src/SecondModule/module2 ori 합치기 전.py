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

    # def get_misconception_text(self, misconception_id: float) -> Optional[str]:
    #     row = self.misconception_df[self.misconception_df['MisconceptionId'] == int(misconception_id)]
    #     if not row.empty:
    #         return row.iloc[0]['MisconceptionName']
    #     logger.warning(f"No misconception found for ID: {misconception_id}")
    #     return None

    def get_misconception_text(self, misconception_id: float) -> Optional[str]:
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
        misconception_clause = (f"that targets the following misconception: \"{misconception_text}\"." if misconception_text != "There is no misconception" else "")
        return f"""
        You are an educational assistant designed to generate multiple-choice questions {misconception_clause}

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
        """

    def call_model_api(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            response.raise_for_status()
            return response.json().get('generated_text', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise

    def parse_model_output(self, output: str) -> GeneratedQuestion:
        output_lines = output.strip().splitlines()
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

    def generate_similar_question_with_text(self, construct_name: str, subject_name: str, question_text: str, correct_answer_text: str, wrong_answer_text: str, misconception_id: float) -> Tuple[Optional[GeneratedQuestion], Optional[str]]:
        misconception_text = self.get_misconception_text(misconception_id)
        if not misconception_text:
            logger.info("Skipping question generation due to lack of misconception.")
            return None, None

        prompt = self.generate_prompt(construct_name, subject_name, question_text, correct_answer_text, wrong_answer_text, misconception_text)

        try:
            generated_text = self.call_model_api(prompt)
            generated_question = self.parse_model_output(generated_text)
            return generated_question, generated_text
        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            return None, None
