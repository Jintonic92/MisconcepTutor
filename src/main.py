import pandas as pd
from module1 import MisconceptionPredictor
from module2 import SimilarQuestionGenerator
from module3 import SelfConsistencyChecker

if __name__ == "__main__":
    # train.csv 로드
    df = pd.read_csv('train_updated.csv')

    # 모듈 초기화
    predictor = MisconceptionPredictor(misconception_csv_path='misconception_mapping.csv')
    generator = SimilarQuestionGenerator(misconception_csv_path='misconception_mapping.csv')
    checker = SelfConsistencyChecker()

    # 앞 10행에 대해 파이프라인 수행
    for idx, row in df.iloc[:10].iterrows():
        construct_name = row['ConstructName']
        subject_name = row['SubjectName']
        question_text = row['QuestionText']
        correct_answer = row['CorrectAnswer'].strip()  # 'A', 'B', 'C', 'D' 중 하나
        correct_answer_text = row[f"Answer{correct_answer}Text"]

        # 원본 문제의 4개 선지 텍스트 (필요 시 비교용)
        original_choices = {
            "A": row["AnswerAText"],
            "B": row["AnswerBText"],
            "C": row["AnswerCText"],
            "D": row["AnswerDText"],
        }

        # 틀린 선지 하나 선택
        possible_answers = ['A', 'B', 'C', 'D']
        wrong_answer_candidates = [ans for ans in possible_answers if ans != correct_answer]
        wrong_answer = wrong_answer_candidates[0]
        wrong_answer_text = row[f"Answer{wrong_answer}Text"]

        # Module1 호출
        misconception_id, misconception_text = predictor.predict_misconception(
            construct_name,
            subject_name,
            question_text,
            correct_answer_text,
            wrong_answer_text,
            wrong_answer,
            row
        )

        print(f"\n[Row {idx}]")
        print("[Module1 Output]")
        print("Misconception Id:", misconception_id)
        print("Misconception Text:", misconception_text)

        # ----------------------------
        #    재생성 루프 (Module2→3)
        # ----------------------------
        mismatch_count = 0
        max_retries = 5

        while True:
            # Module2 호출: 유사 문항 생성
            gen_question, raw_output = generator.generate_similar_question_with_text(
                construct_name=construct_name,
                subject_name=subject_name,
                question_text=question_text,
                correct_answer_text=correct_answer_text,
                wrong_answer_text=wrong_answer_text,
                misconception_id=misconception_id
            )

            if not gen_question:
                print("[Module2 Output] No valid question generated. Skipping Module3.")
                break

            # 출력
            print("\n[Module2 Output] Generated Similar Question:")
            print("Question:", gen_question.question)
            for k, v in gen_question.choices.items():
                print(f"{k}) {v}")
            print("Correct Answer (Gold):", gen_question.correct_answer)
            print("Explanation:", gen_question.explanation)

            # Module3: 10번 추론 → 다수결 결과
            predicted_answer, explanation = checker.check_answer(
                question=gen_question.question,
                choices=gen_question.choices,
                num_inferences=10
            )

            print("\n[Module3 Output] Self-Consistency Check (Majority Vote) Result:")
            print("Predicted Answer:", predicted_answer)
            print("Gold Answer:", gen_question.correct_answer)

            # 비교 (gold answer vs predicted_answer)
            # gold answer가 "A) ..." 처럼 되어 있으면, "A" 부분만 떼어 비교해야 할 수도 있음
            gold_answer_letter = gen_question.correct_answer.split(")")[0].strip()  # "A)" -> "A"

            if predicted_answer.upper() == gold_answer_letter.upper():
                print("=> 정답 일치! 문항 제공을 진행합니다.")
                break
            else:
                mismatch_count += 1
                print("=> 정답 불일치, 문제를 재생성합니다.")
                if mismatch_count >= max_retries:
                    print("재생성 한도 초과! 문항 생성을 중단합니다.")
                    break

        print(f"재생성 횟수(mismatch_count): {mismatch_count}")
        print("--------------------------------------------------------")
