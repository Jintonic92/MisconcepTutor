# module3.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import logging
from config import Llama3_8b_PATH
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SelfConsistencyChecker:
    def __init__(self, model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct'):
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load the language model for self-consistency checking."""
        logger.info(f"Loading model '{model_name}' from '{Llama3_8b_PATH}' for self-consistency check...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=Llama3_8b_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=Llama3_8b_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
            logger.info("Model loaded on GPU for self-consistency.")
        else:
            logger.info("Model loaded on CPU for self-consistency.")

    def _create_prompt(self, question: str, choices: dict) -> str:
        """
        Create a prompt following the Llama 3 prompt template.
        """
        prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an expert reasoning assistant. Your task is to determine the single most accurate answer (A, B, C, or D) for a multiple-choice question based on the given options.

        Rules:
        1. Carefully read the question and all options.
        2. Use logical reasoning to select the best answer.
        3. Output your answer strictly in the following format: "Answer: [A/B/C/D]"
        4. Do not provide any explanation or extra information.

        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Question: {question}

        Choices:
        A) {choices['A']}
        B) {choices['B']}
        C) {choices['C']}
        D) {choices['D']}

        Please select the correct answer.
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        return prompt.strip()

    def _extract_answer(self, text: str) -> str:
        """
        Extract the answer (A, B, C, or D) from the generated text.
        """
        match = re.search(r"Answer:\s*([ABCD])", text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            logger.info(f"Extracted answer: {answer} from text: {text}")
            return answer
        logger.warning(f"Failed to extract answer from text: {text}")
        return ""

    def check_answer(self, question: str, choices: dict, num_inferences: int = 10) -> Tuple[str, str]:
        """
        Perform self-consistency check:
        - Run inference num_inferences times.
        - Extract answer each time.
        - Majority vote the final answer.
        """

        prompt = self._create_prompt(question, choices)  # 수정된 프롬프트 생성
        answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

        inputs = self.tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        for _ in range(num_inferences):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = self._extract_answer(generated_text)

            logger.info(f"Generated text: {generated_text}")  # 모델이 생성한 텍스트 확인
            logger.info(f"Predicted answer: {predicted_answer}")  # 추출된 정답 확인

            if predicted_answer in answer_counts:
                answer_counts[predicted_answer] += 1
            else:
                logger.warning(f"Invalid answer extracted: {predicted_answer}")

        # Majority vote
        final_answer = max(answer_counts, key=answer_counts.get)
        explanation = f"Answer counts: {answer_counts}. Majority answer: {final_answer}"

        logger.info(f"Answer counts: {answer_counts}")
        logger.info(f"Final Answer: {final_answer}")

        return final_answer, explanation
