import streamlit as st

# 앱의 기본 설정
st.set_page_config(
    page_title="MisconcepTutor",
    page_icon="📚",
    layout="wide"
)

# 제목 표시
st.title("MisconcepTutor")

# 기본 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 'welcome'

# 환영 화면
def show_welcome():
    st.write("Smarter learning with customized quizzes!")
    if st.button("퀴즈 시작하기"):
        st.session_state.step = 'question'
        st.rerun()

# 문제 화면
def show_question():
    st.subheader("문제")
    st.write("1 + 2 * 3 = ?")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("A. 9"):
            st.session_state.selected = 'A'
            st.session_state.step = 'result'
            st.rerun()
    with col2:
        if st.button("B. 7"):
            st.session_state.selected = 'B'
            st.session_state.step = 'result'
            st.rerun()
    with col3:
        if st.button("C. 8"):
            st.session_state.selected = 'C'
            st.session_state.step = 'result'
            st.rerun()
    with col4:
        if st.button("D. 5"):
            st.session_state.selected = 'D'
            st.session_state.step = 'result'
            st.rerun()

# 결과 화면
def show_result():
    if st.session_state.selected == 'B':
        st.success("정답입니다!")
    else:
        st.error("틀렸습니다.")
        st.info("연산의 우선순위를 고려해야 합니다.")
        st.write("""
        **설명:**
        1. 2 * 3 = 6 (곱셈 먼저)
        2. 1 + 6 = 7 (그 다음 덧셈)
        """)
    
    if st.button("유사 문제 풀어보기"):
        st.session_state.step = 'similar'
        st.rerun()

# 유사 문제 화면
def show_similar():
    st.subheader("유사 문제")
    st.write("5 - 2 * 3 = ?")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("A. -1"):
            st.success("정답입니다!")
            st.write("""
            **설명:**
            1. 2 * 3 = 6 (곱셈 먼저)
            2. 5 - 6 = -1 (그 다음 뺄셈)
            """)
            if st.button("처음으로"):
                st.session_state.step = 'welcome'
                st.rerun()
    with col2:
        if st.button("B. 9"):
            st.error("다시 시도해보세요.")
    with col3:
        if st.button("C. 6"):
            st.error("다시 시도해보세요.")
    with col4:
        if st.button("D. -6"):
            st.error("다시 시도해보세요.")

# 단계에 따른 화면 표시
if st.session_state.step == 'welcome':
    show_welcome()
elif st.session_state.step == 'question':
    show_question()
elif st.session_state.step == 'result':
    show_result()
elif st.session_state.step == 'similar':
    show_similar()