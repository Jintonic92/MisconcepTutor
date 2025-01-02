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
def load_data(file_name="/train.csv"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    df_folder = os.path.join(base_path, 'Data')
    df = pd.read_csv(df_folder + file_name)
    return df

# misconception mapping 로드
@st.cache_data
def load_misconception_mapping():
    base_path = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(base_path, 'Data', 'misconception_mapping.csv')
    mapping_df = pd.read_csv(mapping_path)
    return mapping_df

def get_misconception_id(question_row, selected_answer):
    """선택한 답변에 해당하는 misconception ID를 반환"""
    misconception_id = None
    if selected_answer == 'A':
        misconception_id = question_row['MisconceptionAId']
    elif selected_answer == 'B':
        misconception_id = question_row['MisconceptionBId']
    elif selected_answer == 'C':
        misconception_id = question_row['MisconceptionCId']
    elif selected_answer == 'D':
        misconception_id = question_row['MisconceptionDId']
    return misconception_id

def get_misconception_description(misconception_id, mapping_df):
    """Misconception ID에 해당하는 설명을 반환"""
    if pd.isna(misconception_id):
        return "Unknown misconception"
    row = mapping_df[mapping_df['MisconceptionId'] == misconception_id]
    if len(row) == 0:
        return "Unknown misconception"
    return row.iloc[0]['MisconceptionName']

@st.cache_resource
def load_question_generator():
    """문제 생성 모델 로드"""
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # 예시 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_similar_question(question, misconception_id, tokenizer, model):
    """
    주어진 문제와 misconception ID를 바탕으로 새로운 문제 생성
    """
    prompt = f"""다음 문제와 misconception을 참고하여 새로운 문제를 생성해주세요:

원본 문제: {question['QuestionText']}
Misconception ID: {misconception_id}

새로운 문제를 생성할 때는 다음 형식을 따라주세요:
- 문제:
- 보기 A:
- 보기 B:
- 보기 C:
- 보기 D:
- 정답:
- 해설:
"""
    
    # 실제 모델을 사용한 생성 로직
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 생성된 텍스트 파싱 (예시)
    return {
        'question': f"[유사 문제] {question['QuestionText']}",
        'choices': {
            'A': f"새로운 보기 A - Misconception ID {misconception_id} 관련",
            'B': f"새로운 보기 B",
            'C': f"새로운 보기 C",
            'D': f"새로운 보기 D"
        },
        'correct': 'A',
        'explanation': f"이 문제는 Misconception ID {misconception_id}와 관련된 개념을 확인하기 위해 생성되었습니다."
    }

def main():
    st.title("MisconcepTutor")
    
    # 데이터 로드
    df = load_data()
    mapping_df = load_misconception_mapping()
    
    # 초기 화면
    if st.session_state.current_step == 'initial':
        st.write("#### 학습을 시작하겠습니다. 10개의 문제가 제공됩니다.")
        if st.button("학습 시작", type="primary"):
            random_questions = df.sample(n=10, random_state=42)
            st.session_state.questions = random_questions
            st.session_state.current_step = 'quiz'
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
                answer = 'A'
                if answer != current_q['CorrectAnswer']:
                    misconception_id = get_misconception_id(current_q, answer)
                    st.session_state.wrong_questions.append(current_q)
                    st.session_state.misconceptions.append(misconception_id)
                st.session_state.current_question_index += 1
                st.rerun()
            
            if st.button(f"C. {current_q['AnswerCText']}", key='C', use_container_width=True):
                answer = 'C'
                if answer != current_q['CorrectAnswer']:
                    misconception_id = get_misconception_id(current_q, answer)
                    st.session_state.wrong_questions.append(current_q)
                    st.session_state.misconceptions.append(misconception_id)
                st.session_state.current_question_index += 1
                st.rerun()

        with col2:
            if st.button(f"B. {current_q['AnswerBText']}", key='B', use_container_width=True):
                answer = 'B'
                if answer != current_q['CorrectAnswer']:
                    misconception_id = get_misconception_id(current_q, answer)
                    st.session_state.wrong_questions.append(current_q)
                    st.session_state.misconceptions.append(misconception_id)
                st.session_state.current_question_index += 1
                st.rerun()
            
            if st.button(f"D. {current_q['AnswerDText']}", key='D', use_container_width=True):
                answer = 'D'
                if answer != current_q['CorrectAnswer']:
                    misconception_id = get_misconception_id(current_q, answer)
                    st.session_state.wrong_questions.append(current_q)
                    st.session_state.misconceptions.append(misconception_id)
                st.session_state.current_question_index += 1
                st.rerun()

        if st.session_state.current_question_index >= 10:
            st.session_state.current_step = 'review'
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
                    
                    misconception_desc = get_misconception_description(misconception_id, mapping_df)
                    st.write("---")
                    st.write("**관련된 Misconception:**")
                    st.info(f"ID: {misconception_id}\n\n{misconception_desc}")
                    
                    if st.button(f"유사 문제 생성하기 #{i+1}"):
                        # 문제 생성 로직
                        tokenizer, model = load_question_generator()
                        new_question = generate_similar_question(wrong_q, misconception_id, tokenizer, model)
                        st.session_state.generated_questions.append(new_question)
                        st.session_state.current_step = f'practice_{i}'
                        st.rerun()

        if st.button("처음으로 돌아가기"):
            for key in st.session_state.keys():
                del st.session_state[key]
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
                if st.button(f"{choice}. {gen_q['choices'][choice]}", use_container_width=True):
                    if choice == gen_q['correct']:
                        st.success("정답입니다!")
                    else:
                        st.error("틀렸습니다. 다시 한 번 풀어보세요.")
                    st.info(gen_q['explanation'])

        with col2:
            for choice in ['B', 'D']:
                if st.button(f"{choice}. {gen_q['choices'][choice]}", use_container_width=True):
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