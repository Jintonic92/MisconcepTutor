import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# Hugging Face 로그인
from huggingface_hub import login
login(token="")
# 모델 불러오기
model_name = "minsuas/Misconceptions__1"
model = SentenceTransformer(model_name)

# 데이터 전처리
def preprocess(df):
    is_train = "MisconceptionAId" in df.columns
    df_new = df.copy()
    
    # 문자열 처리
    for col in df.columns[df.dtypes == "object"]:
        df_new[col] = df_new[col].str.strip()

    # AnswerText 처리
    for option in ["A", "B", "C", "D"]:
        df_new[f"Answer{option}Text"] = df_new[f"Answer{option}Text"].str.replace("Only\n", "Only ")
    
    return df_new

def wide_to_long(df):
    is_train = "MisconceptionAId" in df.columns
    rows = []

    for _, row in df.iterrows():
        correct_option = row["CorrectAnswer"]
        correct_text = row[f"Answer{correct_option}Text"]

        for option in ["A", "B", "C", "D"]:
            if option == correct_option:
                continue
            misconception_id = row[f"Misconception{option}Id"] if is_train else np.nan
            row_new = row[:"QuestionText"]
            row_new["CorrectAnswerText"] = correct_text
            row_new["Answer"] = option
            row_new["AnswerText"] = row[f"Answer{option}Text"]
            if is_train and not np.isnan(misconception_id):
                row_new["MisconceptionId"] = int(misconception_id)
            rows.append(row_new)

    df_long = pd.DataFrame(rows).reset_index(drop=True)
    df_long.insert(0, "QuestionId_Answer", df_long["QuestionId"].astype(str) + "_" + df_long["Answer"])
    df_long = df_long.drop(["Answer", "CorrectAnswer"], axis=1)

    return df_long

# 데이터 불러오기
test_df = pd.read_csv("/content/test.csv")  # 테스트 파일 경로
test_df = preprocess(test_df)
test_df_long = wide_to_long(test_df)

# 쿼리 생성
prompt = (
    "Subject: {SubjectName}\n"
    "Construct: {ConstructName}\n"
    "Question: {QuestionText}\n"
    "Incorrect Answer: {AnswerText}"
)

queries_test = [
    prompt.format(
        SubjectName=row["SubjectName"],
        ConstructName=row["ConstructName"],
        QuestionText=row["QuestionText"],
        AnswerText=row["AnswerText"]
    ) for _, row in test_df_long.iterrows()
]
test_df_long["anchor"] = queries_test

# Misconception 매핑 불러오기
df_map = pd.read_parquet("/content/misconception_mapping.parquet")
sr_map = df_map.set_index("MisconceptionId")["MisconceptionName"]

# 테스트 데이터 임베딩
embs_test_query = model.encode(test_df_long["anchor"], normalize_embeddings=True)

# Misconception 임베딩 불러오기
list_embs_misconception = [np.load("/content/embs_misconception-9-9.npy") for _ in range(len(df_map.columns) - 2)]

# 유사도 계산 및 순위 산출
rank_test = np.array([
    np.argsort(np.argsort(-cosine_similarity(embs_test_query, embs_misconception)), axis=1, kind="stable")
    for embs_misconception in list_embs_misconception
])

# 평균 순위 계산
rank_ave_test = np.mean(rank_test ** (1 / 4), axis=0)
argsort_test = np.argsort(rank_ave_test, axis=1, kind="stable")

# 예측 결과 저장
test_df_long["PredictedMisconceptions"] = [argsort_test[i, :25].tolist() for i in range(len(argsort_test))]

# 예시로 첫 번째 질문의 예측 확인
sample_idx = 2
print("Anchor:", test_df_long.iloc[sample_idx]["anchor"])

top_predictions = argsort_test[sample_idx, :1]  # 상위 1개 예측
print("\nTop 1 Predicted Misconceptions:")
for rank, pred_idx in enumerate(top_predictions, 1):
    print(f"{rank}. {sr_map.iloc[pred_idx]}")
