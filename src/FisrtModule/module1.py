### module1.py
# Misconception을 예측하는 모듈 (나중에 따로 구현 후 그 모델을 불러오는 식으로 구현 할 예정이며, 아직은 mock모듈)
import pandas as pd

class MisconceptionPredictor:
    def __init__(self, misconception_csv_path='misconception_mapping.csv'):
        self.misconception_df = pd.read_csv(misconception_csv_path)
    
    def get_misconception_text(self, misconception_id: int) -> str:
        row = self.misconception_df[self.misconception_df['MisconceptionId'] == misconception_id]
        if not row.empty:
            return row.iloc[0]['MisconceptionName']
        # 해당 id에 대한 misconception이 없으면 기본 텍스트
        return "There is no misconception"
    
    def predict_misconception(self, 
                              construct_name: str, 
                              subject_name: str, 
                              question_text: str, 
                              correct_answer_text: str, 
                              wrong_answer_text: str,
                              wrong_answer: str,
                              row) -> (int, str):
        """
        틀린 선지(wrong_answer)에 해당하는 MisconceptionXId를 row에서 찾고,
        해당 ID의 misconception text를 misconception_mapping에서 찾아 반환.
        """
        # wrong_answer에 따라 MisconceptionXId 컬럼명 결정
        misconception_col = f"Misconception{wrong_answer}Id"
        if misconception_col not in row:
            # 혹시 해당 col이 없으면 기본값
            return -1, "There is no misconception"
        
        misconception_id = row[misconception_col]
        if pd.isna(misconception_id):
            # NaN인 경우 -1 처리
            misconception_id = -1
        else:
            misconception_id = int(misconception_id)
        
        misconception_text = self.get_misconception_text(misconception_id)
        return misconception_id, misconception_text
