# 필요한 라이브러리 임포트
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 경고 메시지 무시
import streamlit as st  # 웹 애플리케이션 프레임워크
import pandas as pd    # 데이터 처리
import numpy as np     # 수치 연산
import random         # 랜덤 기능
from transformers import AutoTokenizer, AutoModelForCausalLM  # 자연어 처리 모델
import torch          # 딥러닝 프레임워크
from dotenv import load_dotenv  # 환경 변수 관리
import os             # 파일 및 경로 처리
import time 

# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="MisconcepTutor",  # 브라우저 탭에 표시될 제목
    layout="wide",                # 페이지 레이아웃 (wide/centered)
    initial_sidebar_state="expanded"  # 사이드바 초기 상태
)

# CSV 데이터 로드 함수
@st.cache_data  # Streamlit 캐싱 데코레이터 (성능 최적화)
def load_data(data_file = '/train.csv'):
    try:
        # Data 폴더에서 train.csv 파일 로드
        base_path = os.path.dirname(os.path.abspath(__file__))        
        data_path = os.path.join(base_path, 'Data')
        df = pd.read_csv(data_path + data_file)
        print(f"{data_file} loaded")
        return df
    
    except FileNotFoundError:
        # 파일이 없는 경우 에러 메시지 표시
        st.error("train.csv 파일을 찾을 수 없습니다.")
        return None

# 세션 상태 초기화 함수
def initialize_session_state():
    """
    Streamlit의 세션 상태 변수들을 초기화하는 함수
    세션 상태는 페이지 새로고침 간에 데이터를 유지하는 데 사용
    """
    if 'started' not in st.session_state:
        st.session_state.started = False  # 퀴즈 시작 여부
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'initial'  # 현재 화면 상태
    if 'questions' not in st.session_state:
        st.session_state.questions = None  # 선택된 문제들
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0  # 현재 문제 인덱스
    if 'wrong_questions' not in st.session_state:
        st.session_state.wrong_questions = []  # 틀린 문제 목록
    if 'misconceptions' not in st.session_state:
        st.session_state.misconceptions = []  # 관련된 misconception 목록
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []  # 생성된 유사 문제 목록

# 퀴즈 시작 함수
def start_quiz():
    """
    퀴즈를 시작하기 위한 초기 설정을 수행하는 함수
    10개의 랜덤 문제를 선택하고 세션 상태를 초기화
    """
    df = load_data()
    if df is not None:
        # 10개의 랜덤 문제 선택
                
        #random_seed = int(time.time()) # 현재 시간을 기반으로 랜덤 시드 생성
        st.session_state.questions = df.sample(n=10, random_state=42) # 🐯 문제 중에 제대로 안나오는 것 있어서 일단 괜찮은 42로 설정 
        #st.session_state.questions = df.sample(n=10, random_state=random_seed)

        # 세션 상태 초기화
        st.session_state.started = True
        st.session_state.current_step = 'quiz'
        st.session_state.current_question_index = 0
        st.session_state.wrong_questions = []
        st.session_state.misconceptions = []
        st.session_state.generated_questions = []

# 답변 처리 함수
def handle_answer(answer, current_q):
    """
    사용자의 답변을 처리하는 함수
    
    Parameters:
        answer (str): 사용자가 선택한 답변 (A, B, C, D)
        current_q (pandas.Series): 현재 문제의 데이터
    """
    # 오답인 경우 처리
    if answer != current_q['CorrectAnswer']:
        st.session_state.wrong_questions.append(current_q)
        # 선택한 답변에 해당하는 misconception ID 찾기
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
    
    # 다음 문제로 이동
    st.session_state.current_question_index += 1
    # 모든 문제를 풀었으면 복습 화면으로 이동
    if st.session_state.current_question_index >= 10:
        st.session_state.current_step = 'review'

# 메인 함수
def main():
    """
    애플리케이션의 메인 로직을 처리하는 함수
    화면 상태에 따라 다른 UI를 표시
    """
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
        
        # 보기 표시 (2x2 그리드 레이아웃)
        col1, col2 = st.columns(2)
        # 왼쪽 열 (A, C 보기)
        with col1:
            if st.button(f"A. {current_q['AnswerAText']}", key='A', use_container_width=True):
                handle_answer('A', current_q)
                st.rerun()
            
            if st.button(f"C. {current_q['AnswerCText']}", key='C', use_container_width=True):
                handle_answer('C', current_q)
                st.rerun()
        # 오른쪽 열 (B, D 보기)
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
        
        # 결과에 따른 메시지 표시
        if len(st.session_state.wrong_questions) == 0:
            st.balloons()  # 축하 효과
            st.success("🎉 축하합니다! 모든 문제를 맞추셨어요!")
            st.markdown("""
            ### 🏆 수학왕이십니다! 
            완벽한 점수를 받으셨네요! 수학적 개념을 정확하게 이해하고 계신 것 같습니다.
            """)
        elif len(st.session_state.wrong_questions) <= 3:
            st.success("잘 하셨어요! 조금만 더 연습하면 완벽할 거예요!")
        else:
            st.info("천천히 개념을 복습해보아요. 연습하다 보면 늘어날 거예요!")
        
        # 새로운 문제 세트 시작 옵션
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 새로운 문제 세트 시작하기", use_container_width=True):
                start_quiz()  # 새로운 퀴즈 세트 시작
                st.rerun()
        
        with col2:
            if st.button("🏠 처음으로 돌아가기", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        # 틀린 문제가 있는 경우 분석 표시
        if len(st.session_state.wrong_questions) > 0:
            st.markdown("---")
            st.write("### ✍️ 틀린 문제 분석")
            # 각 틀린 문제에 대해 분석 정보 표시
            for i, (wrong_q, misconception_id) in enumerate(zip(
                st.session_state.wrong_questions,
                st.session_state.misconceptions
            )):
                with st.expander(f"📝 틀린 문제 #{i+1}"):
                    st.write("**📋 문제:**")
                    st.write(wrong_q['QuestionText'])
                    st.write("**✅ 정답:**", wrong_q['CorrectAnswer'])
                    
                    st.write("---")
                    st.write("**🔍 관련된 Misconception:**")
                    if misconception_id and not pd.isna(misconception_id):
                        st.info(f"Misconception ID: {int(misconception_id)}")
                    else:
                        st.info("Misconception 정보가 없습니다.")
                    
                    # 유사 문제 생성 버튼
                    if st.button(f"📚 유사 문제 풀어보기 #{i+1}", 
                            use_container_width=True):
                        # TODO: 실제 문제 생성 모델 연동
                        new_question = {
                            'question': f"[연습 문제] {wrong_q['QuestionText']}",
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

    # 유사 문제 풀이 화면
    elif st.session_state.current_step.startswith('practice_'):
        practice_idx = int(st.session_state.current_step.split('_')[1])
        gen_q = st.session_state.generated_questions[practice_idx]
        
        st.write("### 유사 문제")
        st.write(gen_q['question'])
        
        # 보기 표시 (2x2 그리드 레이아웃)
        col1, col2 = st.columns(2)
        
        with col1:
            for choice in ['A', 'C']:
                if st.button(f"{choice}. {gen_q['choices'][choice]}", 
                           key=f'practice_{choice}', 
                           use_container_width=True):
                    if choice == gen_q['correct']:
                        st.success("정답입니다!")
                    else:
                        st.error("틀렸습니다. 다시 한 번 풀어보세요.")
                    st.info(gen_q['explanation'])

        with col2:
            for choice in ['B', 'D']:
                if st.button(f"{choice}. {gen_q['choices'][choice]}", 
                           key=f'practice_{choice}', 
                           use_container_width=True):
                    if choice == gen_q['correct']:
                        st.success("정답입니다!")
                    else:
                        st.error("틀렸습니다. 다시 한 번 풀어보세요.")
                    st.info(gen_q['explanation'])

        # 복습 화면으로 돌아가기 버튼
        if st.button("복습 화면으로 돌아가기"):
            st.session_state.current_step = 'review'
            st.rerun()

# 메인 실행
if __name__ == "__main__":
    main()

# random_state 42에서 정답
    # D C A A C
    # A B B B B

