import streamlit as stl
import backend as be

stl.title("교보문고 챗봇입니다📚")
# 세션 상태에 메모리와 채팅 기록 저장
if "memory" not in stl.session_state:  # 메모리가 아직 생성되지 않았는지 확인합니다.
    stl.session_state.memory = be.buff_memory()  # 메모리 초기화
stl.session_state.chat_history = []


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

    chat_response = be.cnvs_chain(
        input_text=input_text, memory=stl.session_state.memory
    )

    # 프런트에 표현
    with stl.chat_message("챗봇"):
        stl.markdown(chat_response)

    # 히스토리 저장
    stl.session_state.chat_history.append({"role": "assistant", "text": chat_response})
