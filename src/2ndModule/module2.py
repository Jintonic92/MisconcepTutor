# module2.py
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
from config import Llama3_8b_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeneratedQuestion:
    question: str
    choices: dict
    correct_answer: str
    explanation: str

class SimilarQuestionGenerator:
    def __init__(self, misconception_csv_path: str = 'misconception_mapping.csv', model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct'):
        """
        Initialize the generator by loading the misconception mapping and the language model.
        """
        self._load_data(misconception_csv_path)
        self._load_model(model_name)

    def _load_data(self, misconception_csv_path: str):
        """Load misconception mapping data."""
        logger.info("Loading misconception mapping...")
        self.misconception_df = pd.read_csv(misconception_csv_path)

    def _load_model(self, model_name: str):
        """Load the language model."""
        logger.info(f"Loading model '{model_name}' from '{Llama3_8b_PATH}'...")
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
            logger.info("Model loaded on GPU.")
        else:
            logger.info("Model loaded on CPU.")

    def get_misconception_text(self, misconception_id: float) -> Optional[str]:
        """Retrieve the misconception text based on the misconception ID."""
        row = self.misconception_df[self.misconception_df['MisconceptionId'] == int(misconception_id)]
        if not row.empty:
            return row.iloc[0]['MisconceptionName']
        logger.warning(f"No misconception found for ID: {misconception_id}")
        return None

    def generate_prompt(self, construct_name: str, subject_name: str, question_text: str, correct_answer_text: str, wrong_answer_text: str, misconception_text: str) -> str:
        """Create a prompt for the language model."""
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

    def parse_model_output(self, output: str) -> GeneratedQuestion:
        """Parse the model's output to extract the question details."""
        output_lines = output.strip().splitlines()
        question, choices, correct_answer, explanation = "", {}, "", ""

        for line in output_lines:
            line = line.strip()
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
                correct_answer = line.split(":",1)[1].strip()
            elif line.lower().startswith("explanation:"):
                explanation = line.split(":",1)[1].strip()

        if not question or len(choices) < 4 or not correct_answer or not explanation:
            logger.warning("Incomplete generated question. Some fields might be missing.")
        return GeneratedQuestion(question, choices, correct_answer, explanation)

    def generate_similar_question_with_text(self, construct_name: str, subject_name: str, question_text: str, correct_answer_text: str, wrong_answer_text: str, misconception_id: float) -> Tuple[Optional[GeneratedQuestion], Optional[str]]:
        """Generate a similar question and return the details."""
        misconception_text = self.get_misconception_text(misconception_id)
        if not misconception_text:
            logger.info("Skipping question generation due to lack of misconception.")
            return None, None

        prompt = self.generate_prompt(construct_name, subject_name, question_text, correct_answer_text, wrong_answer_text, misconception_text)

        inputs = self.tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        assistant_start = generated_text.find("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:
            assistant_start += len("<|start_header_id|>assistant<|end_header_id|>")
            assistant_text = generated_text[assistant_start:].strip()
        else:
            # If no special tokens found, use whole text
            assistant_text = generated_text

        try:
            generated_question = self.parse_model_output(assistant_text)
            logger.info("Successfully generated a similar question.")
            return generated_question, generated_text
        except Exception as e:
            logger.error(f"Failed to parse generated question: {e}")
            return None, generated_text
