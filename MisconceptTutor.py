import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import torch
from dotenv import load_dotenv
import os


# 페이지 설정
st.set_page_config(
    page_title="MisconcepTutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'initial'
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'wrong_questions' not in st.session_state:
    st.session_state.wrong_questions = []
if 'misconceptions' not in st.session_state:
    st.session_state.misconceptions = []
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []

# 데이터 로드
@st.cache_data
def load_data(file_name = "/train.csv"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    df_folder = os.path.join(base_path, 'Data')
    df = pd.read_csv(df_folder + file_name)
    print("train loaded")
    return df

# Misconception 모델 로드
@st.cache_resource
def load_misconception_model():
    model = SentenceTransformer("./model-9")  # 학습된 모델 경로
    return model

# 문제 생성 모델 (예시)
def generate_similar_question(misconception, original_question):
    """
    misconception과 original_question을 기반으로 새로운 문제를 생성하는 함수
    실제로는 더 복잡한 생성 로직이 들어갈 것입니다.
    """
    # 예시로 간단한 템플릿 기반 생성
    new_question = {
        'question': f"[유사 문제] {original_question['QuestionText']}",
        'choices': {
            'A': f"새로운 보기 A",
            'B': f"새로운 보기 B",
            'C': f"새로운 보기 C",
            'D': f"새로운 보기 D"
        },
        'correct': 'A',  # 예시로 A를 정답으로 설정
        'explanation': f"이 문제는 {misconception}와 관련된 개념을 확인하기 위해 생성되었습니다."
    }
    return new_question

def main():
    st.title("MisconcepTutor")
    
    # 데이터 로드
    df = load_data()
    
    # 초기 화면: 10개의 랜덤 문제 선택
    if st.session_state.current_step == 'initial':
        st.write("#### 학습을 시작하겠습니다. 10개의 문제가 제공됩니다.")
        if st.button("학습 시작"):
            # 랜덤하게 10개 문제 선택
            random_questions = df.sample(n=10, random_state=42)
            st.session_state.questions = random_questions
            st.session_state.current_step = 'quiz'
            st.rerun()

    # 퀴즈 화면
    elif st.session_state.current_step == 'quiz':
        current_q = st.session_state.questions.iloc[st.session_state.current_question_index]
        
        st.write(f"### 문제 {st.session_state.current_question_index + 1}/10")
        st.write(current_q['QuestionText'])
        
        # 보기 표시
        answer = None
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"A. {current_q['AnswerAText']}", key='A'):
                answer = 'A'
        with col2:
            if st.button(f"B. {current_q['AnswerBText']}", key='B'):
                answer = 'B'
        col3, col4 = st.columns(2)
        with col3:
            if st.button(f"C. {current_q['AnswerCText']}", key='C'):
                answer = 'C'
        with col4:
            if st.button(f"D. {current_q['AnswerDText']}", key='D'):
                answer = 'D'
                
        if answer:
            # 정답 체크
            if answer != current_q['CorrectAnswer']:
                st.session_state.wrong_questions.append(current_q)
                
                # Misconception 분석
                model = load_misconception_model()
                # TODO: 실제 misconception 분석 로직 구현
                misconception = f"이 문제와 관련된 misconception입니다."
                st.session_state.misconceptions.append(misconception)
                
                # 유사 문제 생성
                new_question = generate_similar_question(misconception, current_q)
                st.session_state.generated_questions.append(new_question)
                
            st.session_state.current_question_index += 1
            
            if st.session_state.current_question_index >= 10:
                st.session_state.current_step = 'review'
            st.rerun()

    # 복습 화면
    elif st.session_state.current_step == 'review':
        st.write("### 학습 결과")
        st.write(f"총 {len(st.session_state.wrong_questions)}개의 문제를 틀렸습니다.")
        
        if len(st.session_state.wrong_questions) > 0:
            st.write("### 오답 노트")
            for i, (wrong_q, misconception, gen_q) in enumerate(zip(
                st.session_state.wrong_questions,
                st.session_state.misconceptions,
                st.session_state.generated_questions
            )):
                st.write(f"#### {i+1}. 틀린 문제")
                st.write(wrong_q['QuestionText'])
                st.write("**Misconception:**")
                st.write(misconception)
                
                st.write("**유사 문제:**")
                st.write(gen_q['question'])
                for choice, text in gen_q['choices'].items():
                    st.write(f"{choice}. {text}")
                
                if st.button(f"유사 문제 풀어보기 #{i+1}"):
                    st.session_state.current_step = f'practice_{i}'
                    st.rerun()

    # 유사 문제 풀이 화면
    elif st.session_state.current_step.startswith('practice_'):
        practice_idx = int(st.session_state.current_step.split('_')[1])
        gen_q = st.session_state.generated_questions[practice_idx]
        
        st.write("### 유사 문제 풀이")
        st.write(gen_q['question'])
        
        for choice, text in gen_q['choices'].items():
            if st.button(f"{choice}. {text}", key=f'practice_{choice}'):
                if choice == gen_q['correct']:
                    st.success("정답입니다!")
                else:
                    st.error("틀렸습니다. 다시 한 번 풀어보세요.")
                st.write(gen_q['explanation'])
                if st.button("복습으로 돌아가기"):
                    st.session_state.current_step = 'review'
                    st.rerun()

if __name__ == "__main__":
    main()