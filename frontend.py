import streamlit as stl
import backend as be

stl.title("교보문고 아동도서 챗봇입니다📚")
# 세션 상태에 메모리와 채팅 기록 저장
if "memory" not in stl.session_state:  # 메모리가 아직 생성되지 않았는지 확인합니다.
    stl.session_state.memory = be.get_memory()  # 메모리 초기화

if 'chat_history' not in stl.session_state: #채팅 기록이 아직 생성되지 않았는지 확인하기
    stl.session_state.chat_history = [] #채팅 기록 초기화하기

if 'vector_index' not in stl.session_state: #벡터 인덱스가 아직 생성되지 않았는지 확인합니다.
    with stl.spinner("Indexing document..."): #이 블록의 코드가 실행되는 동안 스피너를 표시합니다.
        stl.session_state.vector_index = be.get_index() #지원 라이브러리를 통해 인덱스를 검색하고 앱의 세션 캐시에 저장합니다.


for message in stl.session_state.chat_history:
    with stl.chat_message(message["role"]):
        stl.markdown(message["text"])

input_text = stl.chat_input("질문을 입력하세요.")

if input_text:
    # 프런트에 표현
    with stl.chat_message("나"):
        stl.markdown(input_text)

    # 히스토리 저장
    stl.session_state.chat_history.append({"role": "user", "text": input_text})

    chat_response = be.get_rag_chat_response(
        input_text=input_text, memory=stl.session_state.memory,index=stl.session_state.vector_index,)

    # 프런트에 표현
    with stl.chat_message("챗봇"):
        stl.markdown(chat_response)

    # 히스토리 저장
    stl.session_state.chat_history.append({"role": "assistant", "text": chat_response})
