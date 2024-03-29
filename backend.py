import os
from langchain.llms import bedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 다른 모듈과 같이 맞물려 동작할 수 있게 해주는 모듈


# 후에 시스템 프롬프트를 구성
# kwargs 추가...
def bedrock_chatbot():
    llm = BedrockChat(
        credentials_profile_name="poc",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0.5},
    )

    return llm


def buff_memory():
    # 입력과 출력을 저장
    buff_memory = bedrock_chatbot()
    memory = ConversationBufferMemory(llm=buff_memory)
    return memory


def cnvs_chain(input_text, memory):
    chain_data = bedrock_chatbot()
    cnvs_chain = ConversationChain(llm=chain_data, memory=memory, verbose=True)
    # 모델 모듈과 메모리 모듈을 엮어서 동작시킴
    chat_reply = cnvs_chain.predict(input=input_text)
    return chat_reply
