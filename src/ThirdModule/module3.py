import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import logging
from config import Llama3_8b_PATH
import re
from collections import Counter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SelfConsistencyChecker:
    def __init__(self, model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct'):
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load the language model for self-consistency checking."""
        logger.info(f"Loading model '{model_name}' from '{Llama3_8b_PATH}' for self-consistency check...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=Llama3_8b_PATH,
            trust_remote_code=True
        )
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
        모델이 오직 최종 정답만 출력하도록 유도하는 프롬프트를 생성.
        (체인오브소트/추가설명은 내부적으로만, 최종 출력은 "Answer: X" 형식으로 제한)
        
        이 예시 prompt는 "Reason carefully but provide only final answer" 컨셉의 구조입니다.
        """
        system_prompt = (
            "You are an expert solution checker for multiple-choice questions. "
            "You will be given a question and four choices (A, B, C, D). "
            "Your job is to determine the single best answer. "
            "You must reason step by step internally (chain-of-thought) but "
            "DO NOT reveal your reasoning. "
            "Output ONLY the final answer in the format: 'Answer: X' "
            "with no extra text."
        )

        # 유저 질문은 보통 <|start_header_id|>user<|end_header_id|> 구간에 포함
        user_prompt = f"""
            Question: {question}

            Choices:
            A) {choices['A']}
            B) {choices['B']}
            C) {choices['C']}
            D) {choices['D']}

            Select one correct option from A, B, C, or D.
            """.strip()

        # Llama/Chat류 모델에서 system vs. user 구분을 해줄 수도 있음
        # 여기서는 간단히 "<|begin_of_text|>" ~ 같은 태그를 붙이거나,
        # 또는 Chat형식의 meta-prompt를 구성할 수 있습니다.
        combined_prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
                    """.strip()

        return combined_prompt

    def _extract_answer(self, text: str) -> str:
        """
        "Answer: X"에서 X가 A/B/C/D인지 추출
        """
        match = re.search(r"Answer:\s*([ABCD])", text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            logger.debug(f"Extracted answer: {answer} from text: {text}")
            return answer
        logger.warning(f"Failed to extract answer from text: {text}")
        return ""

    def check_answer(self, question: str, choices: dict, num_inferences: int = 10) -> Tuple[str, str]:
        """
        1) 동일 질문에 대해 num_inferences번 반복 추론
        2) 각각 "Answer: X" 형태를 파싱
        3) 최빈값(majority vote)을 최종 답으로 결정
        4) explanation에는 debug용으로 전체 투표 결과를 간단 출력
        """

        # 우선 프롬프트 생성
        prompt = self._create_prompt(question, choices)

        # tokenizer 인풋
        inputs = self.tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # 여러 번(=num_inferences) 추론
        answers = []
        for _ in range(num_inferences):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    # do_sample=True 이면 랜덤성 높아짐.
                    # 다수결을 시험하기 위해 일단 do_sample=True 유지 가능.
                    # 더 일관된 결과를 원한다면 do_sample=False, temperature=0 등으로 바꿀 수도 있음.
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            extracted = self._extract_answer(generated_text)
            if extracted in ["A", "B", "C", "D"]:
                answers.append(extracted)

        if not answers:
            # 아무 답도 추출 못했다면 fallback
            return "", "No valid answers extracted."

        # 다수결
        counter = Counter(answers)
        final_answer = counter.most_common(1)[0][0]  # 가장 많이 나온 1개
        explanation = f"All answers: {answers}, counts: {dict(counter)}, final: {final_answer}"

        return final_answer, explanation
