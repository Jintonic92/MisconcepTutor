import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

# 페이지 설정
st.set_page_config(
    page_title="MisconcepTutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터 로드
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Data/train.csv")
        return df
    except FileNotFoundError:
        st.error("train.csv 파일을 찾을 수 없습니다.")
        return None

def initialize_session_state():
    if 'started' not in st.session_state:
        st.session_state.started = False
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

def start_quiz():
    df = load_data()
    if df is not None:
        st.session_state.questions = df.sample(n=10, random_state=42)
        st.session_state.started = True
        st.session_state.current_step = 'quiz'
        st.session_state.current_question_index = 0
        st.session_state.wrong_questions = []
        st.session_state.misconceptions = []
        st.session_state.generated_questions = []

def handle_answer(answer, current_q):
    if answer != current_q['CorrectAnswer']:
        st.session_state.wrong_questions.append(current_q)
        misconception_id = None
        if answer == 'A':
            misconception_id = current_q.get('MisconceptionAId')
        elif answer == 'B':
            misconception_id = current_q.get('MisconceptionBId')
        elif answer == 'C':
            misconception_id = current_q.get('MisconceptionCId')
        elif answer == 'D':
            misconception_id = current_q.get('MisconceptionDId')
        st.session_state.misconceptions.append(misconception_id)
    
    st.session_state.current_question_index += 1
    if st.session_state.current_question_index >= 10:
        st.session_state.current_step = 'review'

def main():
    st.title("MisconcepTutor")
    
    initialize_session_state()
    
    # 초기 화면
    if st.session_state.current_step == 'initial':
        st.write("#### 학습을 시작하겠습니다. 10개의 문제가 제공됩니다.")
        if st.button("학습 시작", type="primary"):
            start_quiz()
            st.rerun()

    # 퀴즈 화면
    elif st.session_state.current_step == 'quiz':
        current_q = st.session_state.questions.iloc[st.session_state.current_question_index]
        
        # 진행 상황 표시
        progress = st.session_state.current_question_index / 10
        st.progress(progress)
        st.write(f"### 문제 {st.session_state.current_question_index + 1}/10")
        
        # 문제 표시
        st.markdown("---")
        st.write(current_q['QuestionText'])
        
        # 보기 표시
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"A. {current_q['AnswerAText']}", key='A', use_container_width=True):
                handle_answer('A', current_q)
                st.rerun()
            
            if st.button(f"C. {current_q['AnswerCText']}", key='C', use_container_width=True):
                handle_answer('C', current_q)
                st.rerun()

        with col2:
            if st.button(f"B. {current_q['AnswerBText']}", key='B', use_container_width=True):
                handle_answer('B', current_q)
                st.rerun()
            
            if st.button(f"D. {current_q['AnswerDText']}", key='D', use_container_width=True):
                handle_answer('D', current_q)
                st.rerun()

    # 복습 화면
    elif st.session_state.current_step == 'review':
        st.write("### 학습 결과")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 문제 수", "10")
        with col2:
            st.metric("맞은 문제", f"{10 - len(st.session_state.wrong_questions)}")
        with col3:
            st.metric("틀린 문제", f"{len(st.session_state.wrong_questions)}")
        
        if len(st.session_state.wrong_questions) > 0:
            st.write("### 틀린 문제 분석")
            for i, (wrong_q, misconception_id) in enumerate(zip(
                st.session_state.wrong_questions,
                st.session_state.misconceptions
            )):
                with st.expander(f"틀린 문제 #{i+1}"):
                    st.write("**문제:**")
                    st.write(wrong_q['QuestionText'])
                    st.write("**정답:**", wrong_q['CorrectAnswer'])
                    
                    st.write("---")
                    st.write("**관련된 Misconception:**")
                    if misconception_id and not pd.isna(misconception_id):
                        st.info(f"Misconception ID: {int(misconception_id)}")
                    else:
                        st.info("Misconception 정보가 없습니다.")
                    
                    if st.button(f"유사 문제 생성하기 #{i+1}"):
                        # 임시 문제 생성 로직
                        new_question = {
                            'question': f"[유사 문제] {wrong_q['QuestionText']}",
                            'choices': {
                                'A': "새로운 보기 A",
                                'B': "새로운 보기 B",
                                'C': "새로운 보기 C",
                                'D': "새로운 보기 D"
                            },
                            'correct': 'A',
                            'explanation': f"이 문제는 Misconception ID {misconception_id}와 관련되어 있습니다."
                        }
                        st.session_state.generated_questions.append(new_question)
                        st.session_state.current_step = f'practice_{i}'
                        st.rerun()

        if st.button("처음으로 돌아가기"):
            st.session_state.clear()
            st.rerun()

    # 유사 문제 풀이 화면
    elif st.session_state.current_step.startswith('practice_'):
        practice_idx = int(st.session_state.current_step.split('_')[1])
        gen_q = st.session_state.generated_questions[practice_idx]
        
        st.write("### 유사 문제")
        st.write(gen_q['question'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            for choice in ['A', 'C']:
                if st.button(f"{choice}. {gen_q['choices'][choice]}", key=f'practice_{choice}', use_container_width=True):
                    if choice == gen_q['correct']:
                        st.success("정답입니다!")
                    else:
                        st.error("틀렸습니다. 다시 한 번 풀어보세요.")
                    st.info(gen_q['explanation'])

        with col2:
            for choice in ['B', 'D']:
                if st.button(f"{choice}. {gen_q['choices'][choice]}", key=f'practice_{choice}', use_container_width=True):
                    if choice == gen_q['correct']:
                        st.success("정답입니다!")
                    else:
                        st.error("틀렸습니다. 다시 한 번 풀어보세요.")
                    st.info(gen_q['explanation'])

        if st.button("복습 화면으로 돌아가기"):
            st.session_state.current_step = 'review'
            st.rerun()

if __name__ == "__main__":
    main()