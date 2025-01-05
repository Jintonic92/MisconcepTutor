import streamlit as st
import pandas as pd
import os
from src.SecondModule.module2 import SimilarQuestionGenerator
import logging
from typing import Optional, Tuple
logging.basicConfig(level=logging.DEBUG)


# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="MisconcepTutor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 경로 설정
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')
misconception_csv_path = os.path.join(data_path, 'misconception_mapping.csv')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 세션 상태 초기화 - 가장 먼저 실행되도록 최상단에 배치
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.wrong_questions = []
    st.session_state.misconceptions = []
    st.session_state.current_question_index = 0
    st.session_state.generated_questions = []
    st.session_state.current_step = 'initial'
    st.session_state.selected_wrong_answer = None
    st.session_state.questions = []
    logger.info("Session state initialized")

# 문제 생성기 초기화
@st.cache_resource
def load_question_generator():
    """문제 생성 모델 로드"""
    if not os.path.exists(misconception_csv_path):
        st.error(f"CSV 파일이 존재하지 않습니다: {misconception_csv_path}")
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {misconception_csv_path}")
    return SimilarQuestionGenerator(misconception_csv_path=misconception_csv_path)

# CSV 데이터 로드 함수
@st.cache_data
def load_data(data_file = '/train.csv'):
    try:
        file_path = os.path.join(data_path, data_file.lstrip('/'))
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {data_file}")
        logger.error(f"File not found: {data_file}")
        return None

def start_quiz():
    """퀴즈 시작 및 초기화"""
    df = load_data()
    if df is None or df.empty:
        st.error("데이터를 불러올 수 없습니다. 데이터셋을 확인해주세요.")
        return

    st.session_state.questions = df.sample(n=10, random_state=42)
    st.session_state.current_step = 'quiz'
    st.session_state.current_question_index = 0
    st.session_state.wrong_questions = []
    st.session_state.misconceptions = []
    st.session_state.generated_questions = []
    logger.info("Quiz started")


def generate_similar_question(wrong_q, misconception_id, generator):
    """유사 문제 생성"""
    logger.info(f"Generating similar question for misconception_id: {misconception_id}")
    
    # 입력 데이터 유효성 검사
    if not isinstance(wrong_q, dict):
        logger.error(f"Invalid wrong_q type: {type(wrong_q)}")
        st.error("유사 문제 생성에 필요한 데이터 형식이 잘못되었습니다.")
        return None
        
    # misconception_id가 유효한지 확인
    if pd.isna(misconception_id):
        logger.warning("misconception_id is NaN")
        return None
        
    try:
        # 데이터 준비 (튜플 변환 방지)
        input_data = {
            'construct_name': str(wrong_q.get('ConstructName', '')),
            'subject_name': str(wrong_q.get('SubjectName', '')),
            'question_text': str(wrong_q.get('QuestionText', '')),
            'correct_answer_text': str(wrong_q.get(f'Answer{wrong_q["CorrectAnswer"]}Text', '')),
            'wrong_answer_text': str(wrong_q.get(f'Answer{st.session_state.selected_wrong_answer}Text', '')),
            'misconception_id': int(misconception_id)
        }
        
        logger.info(f"Prepared input data: {input_data}")
        
        # 유사 문제 생성 호출
        generated_q, _ = generator.generate_similar_question_with_text(
            construct_name=input_data['construct_name'],
            subject_name=input_data['subject_name'],
            question_text=input_data['question_text'],
            correct_answer_text=input_data['correct_answer_text'],
            wrong_answer_text=input_data['wrong_answer_text'],
            misconception_id=input_data['misconception_id']
        )
        
        if generated_q:
            return {
                'question': generated_q.question,
                'choices': generated_q.choices,
                'correct': generated_q.correct_answer,
                'explanation': generated_q.explanation
            }
            
    except Exception as e:
        logger.error(f"Error in generate_similar_question: {str(e)}")
        st.error(f"문제 생성 중 오류가 발생했습니다: {str(e)}")
        return None

    return None

def handle_answer(answer, current_q):
    """답변 처리"""
    if answer != current_q['CorrectAnswer']:
        wrong_q_dict = current_q.to_dict()
        st.session_state.wrong_questions.append(wrong_q_dict)
        st.session_state.selected_wrong_answer = answer
        
        misconception_key = f'Misconception{answer}Id'
        misconception_id = current_q.get(misconception_key)
        st.session_state.misconceptions.append(misconception_id)
    
    st.session_state.current_question_index += 1
    if st.session_state.current_question_index >= 10:
        st.session_state.current_step = 'review'

def main():
    """메인 애플리케이션 로직"""
    st.title("MisconcepTutor")
    
    # Generator 초기화
    generator = load_question_generator()

    # 초기 화면
    if st.session_state.current_step == 'initial':
        st.write("#### 학습을 시작하겠습니다. 10개의 문제를 풀어볼까요?")
        if st.button("학습 시작", key="start_quiz"):
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
            if st.button(f"A) {current_q['AnswerAText']}", key="A"):
                handle_answer('A', current_q)
                st.rerun()
            if st.button(f"C) {current_q['AnswerCText']}", key="C"):
                handle_answer('C', current_q)
                st.rerun()
        with col2:
            if st.button(f"B) {current_q['AnswerBText']}", key="B"):
                handle_answer('B', current_q)
                st.rerun()
            if st.button(f"D) {current_q['AnswerDText']}", key="D"):
                handle_answer('D', current_q)
                st.rerun()

    # 복습 화면
    elif st.session_state.current_step == 'review':
        st.write("### 학습 결과")
        
        # 결과 통계
        col1, col2, col3 = st.columns(3)
        col1.metric("총 문제 수", 10)
        col2.metric("맞은 문제", 10 - len(st.session_state.wrong_questions))
        col3.metric("틀린 문제", len(st.session_state.wrong_questions))
        
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
        
        # 네비게이션 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 새로운 문제 세트 시작하기", use_container_width=True):
                start_quiz()
                st.rerun()
        with col2:
            if st.button("🏠 처음으로 돌아가기", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        # 틀린 문제 분석
        if st.session_state.wrong_questions:
            st.write("### ✍️ 틀린 문제 분석")
            for i, (wrong_q, misconception_id) in enumerate(zip(
                st.session_state.wrong_questions,
                st.session_state.misconceptions
            )):
                with st.expander(f"📝 틀린 문제 #{i + 1}"):
                    st.write("**📋 문제:**")
                    st.write(wrong_q['QuestionText'])
                    st.write("**✅ 정답:**", wrong_q['CorrectAnswer'])
                    
                    st.write("---")
                    st.write("**🔍 관련된 Misconception:**")
                    if misconception_id and not pd.isna(misconception_id):
                        misconception_text = generator.get_misconception_text(misconception_id)
                        st.info(f"Misconception ID: {int(misconception_id)}\n\n{misconception_text}")
                    else:
                        st.info("Misconception 정보가 없습니다.")

                    # 틀린 문제 분석 부분에서
                    if st.button(f"📚 유사 문제 풀기 #{i + 1}", key=f"retry_{i}"):
                        # 유사 문제 생성 상태를 세션에 저장
                        st.session_state[f"show_similar_question_{i}"] = True
                        st.session_state[f"similar_question_answered_{i}"] = False
                        # 기존 답변 기록 초기화
                        st.rerun()
                    
                    # 유사 문제가 생성된 상태인 경우
                    if st.session_state.get(f"show_similar_question_{i}", False):
                        with st.spinner("유사 문제를 생성하고 있습니다..."):
                            new_question = generate_similar_question(wrong_q, misconception_id, generator)
                            if new_question:
                                st.write("### 🎯 유사 문제")
                                st.write(new_question['question'])
                                
                                # 답변 상태 확인
                                answered = st.session_state.get(f"similar_question_answered_{i}", False)
                                selected_answer = st.session_state.get(f"selected_answer_{i}", None)
                                
                                # 보기 표시
                                st.write("**보기:**")
                                col1, col2 = st.columns(2)
                                
                                # 답변하지 않은 경우에만 버튼 활성화
                                if not answered:
                                    with col1:
                                        for option in ['A', 'C']:
                                            if st.button(
                                                f"{option}) {new_question['choices'][option]}", 
                                                key=f"similar_{option}_{i}"
                                            ):
                                                st.session_state[f"similar_question_answered_{i}"] = True
                                                st.session_state[f"selected_answer_{i}"] = option
                                                st.rerun()
                                    
                                    with col2:
                                        for option in ['B', 'D']:
                                            if st.button(
                                                f"{option}) {new_question['choices'][option]}", 
                                                key=f"similar_{option}_{i}"
                                            ):
                                                st.session_state[f"similar_question_answered_{i}"] = True
                                                st.session_state[f"selected_answer_{i}"] = option
                                                st.rerun()
                                
                                # 답변한 경우 결과 표시
                                if answered:
                                    selected = st.session_state[f"selected_answer_{i}"]
                                    if selected == new_question['correct']:
                                        st.success("✅ 정답입니다!")
                                    else:
                                        st.error(f"❌ 틀렸습니다. 정답은 {new_question['correct']}입니다.")
                                    
                                    # 해설 표시
                                    st.write("---")
                                    st.write("**📝 해설:**", new_question['explanation'])
                                    
                                    # 다시 풀기 버튼
                                    if st.button("🔄 다시 풀기", key=f"reset_{i}"):
                                        st.session_state[f"similar_question_answered_{i}"] = False
                                        st.session_state[f"selected_answer_{i}"] = None
                                        st.rerun()
                                
                                # 문제 닫기 버튼
                                if st.button("❌ 문제 닫기", key=f"close_{i}"):
                                    st.session_state[f"show_similar_question_{i}"] = False
                                    st.rerun()
                            else:
                                st.error("유사 문제를 생성할 수 없습니다.")
                                if st.button("❌ 닫기", key=f"close_error_{i}"):
                                    st.session_state[f"show_similar_question_{i}"] = False
                                    st.rerun()
if __name__ == "__main__":
    main()

# random_state 42에서 정답
    # D C A A C
    # A B B B B
